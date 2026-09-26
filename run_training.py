from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.modeling.dataset import (
    EVAL_SETS,
    TEST_SET,
    TRAIN_SET,
    VAL_SET,
    load_splits,
)
from src.modeling.folds import check_folds, fold_summary, walk_forward_folds
from src.modeling.metrics import DEFAULT_THRESHOLD
from src.modeling.models import SEED, build_candidates
from src.modeling.train import comparison_record, neg_pos_ratio, run_candidates

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MODELING_REPORTS = REPORTS_DIR / "modeling"
SPLIT_FILE = "split.parquet"
COMPARISON_CSV = "comparison.csv"
COMPARISON_JSON = "comparison.json"

REPORT_COLUMNS = [
    "modelo",
    "familia",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "predict_rate",
    "tn",
    "fp",
    "fn",
    "tp",
    "n_train",
    "n_valid",
    "fit_s",
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

    print("[1/5] Cargando split temporal (sin test) ...")
    digest = _file_digest(split_file)
    splits = load_splits(split_file, names=EVAL_SETS)
    train, valid = splits[TRAIN_SET], splits[VAL_SET]
    print(f"  digest split.parquet: {digest[:16]}...")
    for s in (train, valid):
        print(
            f"  {s.name}: n={s.n_rows} features={s.X.shape[1]} "
            f"fraude={100 * s.pos_rate:.3f}%"
        )
    assert TEST_SET not in splits, "El test no debe cargarse en la Fase G"

    print("[2/5] Definiendo pliegues temporales internos de train ...")
    folds = walk_forward_folds(train.X["TransactionDT"])
    checks = check_folds(train.X["TransactionDT"], folds)
    for c in checks:
        print(f"  {'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    assert all(c["pass"] for c in checks), "La validacion de pliegues fallo"
    print(fold_summary(train.X["TransactionDT"], train.y, folds).to_string(index=False))

    print("[3/5] Entrenando y comparando candidatos (train -> validation) ...")
    pos_weight = neg_pos_ratio(train.y)
    print(f"  desbalance train neg/pos={pos_weight:.2f}")
    candidates = build_candidates(seed=SEED, pos_weight=pos_weight)
    for cand in candidates:
        print(f"  - {cand.family:9s} {cand.name}")
    table, results = run_candidates(
        candidates, train, valid, threshold=DEFAULT_THRESHOLD, seed=SEED
    )
    print()
    print(table[REPORT_COLUMNS].to_string(index=False))

    print("[4/5] Guardando resultados ...")
    MODELING_REPORTS.mkdir(parents=True, exist_ok=True)
    csv_file = MODELING_REPORTS / COMPARISON_CSV
    json_file = MODELING_REPORTS / COMPARISON_JSON
    table[REPORT_COLUMNS].to_csv(csv_file, index=False)

    record = comparison_record(results, seed=SEED, threshold=DEFAULT_THRESHOLD)
    record["split_file"] = SPLIT_FILE
    record["split_sha256"] = digest
    record["n_features"] = int(train.X.shape[1])
    record["pos_weight_train"] = round(pos_weight, 4)
    record["folds"] = json.loads(
        fold_summary(train.X["TransactionDT"], train.y, folds).to_json(orient="records")
    )
    json_file.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  comparacion -> {csv_file}")
    print(f"  registro    -> {json_file}")

    print("[5/5] Resumen por familia ...")
    best = table.iloc[0]
    print(
        f"  mejor por PR-AUC: {best['modelo']} "
        f"pr_auc={best['pr_auc']:.4f} roc_auc={best['roc_auc']:.4f} "
        f"fit={best['fit_s']}s"
    )
    for family, group in table.groupby("familia", sort=False):
        top = group.sort_values("pr_auc", ascending=False).iloc[0]
        print(
            f"  {family:9s} mejor={top['modelo']:17s} "
            f"pr_auc={top['pr_auc']:.4f} roc_auc={top['roc_auc']:.4f}"
        )
    print(
        f"  test intacto (no cargado). Umbral operativo: Fase I. "
        f"Umbral de referencia usado: {DEFAULT_THRESHOLD}"
    )


if __name__ == "__main__":
    main()
