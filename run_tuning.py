from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib

from src.modeling.dataset import EVAL_SETS, TEST_SET, TRAIN_SET, VAL_SET, load_splits
from src.modeling.folds import check_folds, walk_forward_folds
from src.modeling.models import SEED
from src.modeling.space import SEARCH_SPACE, sample_configs
from src.modeling.tuning import confirm, fit_final, run_search, select_best

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
TUNING_REPORTS = REPORTS_DIR / "tuning"
ARTIFACTS_DIR = ROOT / "artifacts"

SPLIT_FILE = "split.parquet"
SEARCH_CSV = "search.csv"
SEARCH_JSON = "search.json"
SELECTED_JSON = "selected.json"
FINAL_MODEL = "model_final.joblib"

N_CONFIGS = 20

SEARCH_COLUMNS = [
    "config_id",
    "mean_pr_auc",
    "std_pr_auc",
    "min_pr_auc",
    "mean_roc_auc",
    "mean_recall",
    "best_iteration",
    "seconds",
    "is_anchor",
    *SEARCH_SPACE.keys(),
]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    split_file = PROCESSED_DIR / SPLIT_FILE
    if not split_file.exists():
        raise FileNotFoundError(
            f"Falta {split_file}. Ejecuta antes: run_preprocessing.py, "
            "run_feature_engineering.py, run_split.py"
        )

    TUNING_REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    search_csv = TUNING_REPORTS / SEARCH_CSV
    search_json = TUNING_REPORTS / SEARCH_JSON
    selected_json = TUNING_REPORTS / SELECTED_JSON

    print("[1/6] Cargando train y validation (test NO se carga) ...")
    digest = _file_digest(split_file)
    splits = load_splits(split_file, names=EVAL_SETS)
    train, valid = splits[TRAIN_SET], splits[VAL_SET]
    assert TEST_SET not in splits, "El test no debe cargarse durante el tuning"
    print(f"  digest split.parquet: {digest[:16]}...")
    for s in (train, valid):
        print(
            f"  {s.name}: n={s.n_rows} features={s.X.shape[1]} "
            f"fraude={100 * s.pos_rate:.3f}%"
        )

    print("[2/6] Verificando pliegues walk-forward internos de train ...")
    folds = walk_forward_folds(train.X["TransactionDT"])
    for c in check_folds(train.X["TransactionDT"], folds):
        print(f"  {'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    assert all(c["pass"] for c in check_folds(train.X["TransactionDT"], folds))

    print(
        f"[3/6] Muestreando {N_CONFIGS} configuraciones (random search, seed={SEED}) ..."
    )
    configs = sample_configs(N_CONFIGS, seed=SEED)
    print(f"  config 0 = ancla de la Fase G: {configs[0]}")
    print("  espacio: " + ", ".join(f"{k}={len(v)}" for k, v in SEARCH_SPACE.items()))

    print("[4/6] Buscando (3 pliegues x config, early stopping sobre PR-AUC) ...")
    started = time.perf_counter()

    def _progress(result: dict[str, Any], done: int, total: int) -> None:
        print(
            f"  [{done:2d}/{total}] pr_auc={result['mean_pr_auc']:.4f}"
            f"+-{result['std_pr_auc']:.4f} roc_auc={result['mean_roc_auc']:.4f}"
            f" iters={result['best_iteration']:4d} {result['seconds']:6.1f}s"
            f" anchor={result['is_anchor']}"
        )

    table, results = run_search(configs, train, folds, seed=SEED, on_result=_progress)
    print(f"  busqueda completada en {(time.perf_counter() - started) / 60:.1f} min")
    table.to_csv(search_csv, index=False)
    print()
    print(table[SEARCH_COLUMNS].to_string(index=False))

    print("[5/6] Seleccion y ajuste final sobre train ...")
    best = select_best(table)
    print(f"  ganadora: config_id={best['config_id']} (anchor={best['is_anchor']})")
    print(f"  params  : {best['params']}")
    print(
        f"  rondas  : {best['best_iteration']} (mediana del early stopping por pliegue)"
    )
    print(
        f"  pliegues: pr_auc={best['mean_pr_auc']:.4f} roc_auc={best['mean_roc_auc']:.4f}"
    )
    estimator = fit_final(best["params"], best["best_iteration"], train, seed=SEED)

    print("[6/6] Confirmando en validation y guardando ...")
    valid_metrics = confirm(estimator, valid)
    print(
        f"  validation: pr_auc={valid_metrics['pr_auc']:.4f} "
        f"roc_auc={valid_metrics['roc_auc']:.4f} "
        f"recall@0.5={valid_metrics['recall']:.4f} "
        f"precision@0.5={valid_metrics['precision']:.4f}"
    )
    model_file = ARTIFACTS_DIR / FINAL_MODEL
    joblib.dump(estimator, model_file)
    print(f"  modelo -> {model_file} ({model_file.stat().st_size / 1e6:.1f} MB)")

    record: dict[str, Any] = {
        "seed": SEED,
        "n_configs": N_CONFIGS,
        "split_file": SPLIT_FILE,
        "split_sha256": digest,
        "n_features": int(train.X.shape[1]),
        "folds": len(folds),
        "criterio_seleccion": "PR-AUC medio de los pliegues walk-forward de train",
        "desempate": "ROC-AUC medio",
        "test_usado": False,
        "entrenado_en": TRAIN_SET,
        "confirmado_en": VAL_SET,
        "seleccion": best,
        "validacion": valid_metrics,
        "modelo": {
            "artefacto": FINAL_MODEL,
            "clase": type(estimator).__name__,
            "n_rondas": int(estimator.n_estimators),
        },
        "busqueda": [{k: v for k, v in r.items() if k != "folds"} for r in results],
    }
    search_json.write_text(
        json.dumps({**record, "busqueda": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected_json.write_text(
        json.dumps(
            {k: v for k, v in record.items() if k != "busqueda"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  busqueda -> {search_csv}")
    print(f"  registro  -> {search_json}")
    print(f"  seleccion -> {selected_json}")
    print(
        "  El test sigue sin consumirse: se evalua una sola vez en la Fase I (T-I01)."
    )


if __name__ == "__main__":
    main()
