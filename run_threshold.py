"""Fija el umbral operativo ANTES de consumir `test`.

Este script es la unica via para calcular el umbral y **no carga `test`**. El
umbral sale de `validation`, que es el unico bloque fuera de muestra puntuado
por el artefacto exacto que se despliega; las predicciones out-of-fold de
`train` se calculan aparte como corroboracion independiente.

Motivo de calibrar en `validation` y no en `train` out-of-fold: un umbral
fijado sobre out-of-fold NO traslado. Medido en la Fase I, el umbral que daba
recall 0.8001 en `train` out-offold daba solo 0.7763 en `validation`. Los
modelos out-of-fold se entrenan con 234k-369k filas y el final con 434k, de
modo que el final reparte sus scores de forma mas apretada: a igual umbral
bajan las alertas (13.5% -> 11.4%) y con ellas el recall, mientras la
precision no se mueve. Calibrar sobre la distribucion de scores que realmente
se despliega evita ese sesgo; `test` queda como unica estimacion limpia.

Escribir `reports/evaluation/threshold.json` es la precondicion que habilita
`run_evaluation.py`, de modo que el umbral queda fijado antes de que exista
cualquier numero sobre el conjunto de test.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.modeling.dataset import EVAL_SETS, TEST_SET, TRAIN_SET, VAL_SET, load_splits
from src.modeling.evaluate import (
    TARGET_RECALL,
    out_of_fold_scores,
    recall_interval,
    threshold_for_target_recall,
    threshold_sweep,
)
from src.modeling.folds import check_folds, walk_forward_folds
from src.modeling.metrics import summarize
from src.modeling.models import SEED

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
EVAL_REPORTS = REPORTS_DIR / "evaluation"

SPLIT_FILE = "split.parquet"
TUNING_DIR = REPORTS_DIR / "tuning"
SELECTED_JSON = TUNING_DIR / "selected.json"
THRESHOLD_JSON = "threshold.json"
SWEEP_CSV = "threshold_sweep.csv"
OOF_SWEEP_CSV = "oof_sweep.csv"


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
    if not SELECTED_JSON.exists():
        raise FileNotFoundError(
            f"Falta {SELECTED_JSON}. El umbral se fija sobre el modelo que "
            "eligio la Fase H; ejecuta antes: run_tuning.py"
        )

    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    if selected.get("test_usado") is not False:
        raise ValueError("selected.json no certifica test_usado=false")
    digest = _file_digest(split_file)
    if digest != selected["split_sha256"]:
        raise ValueError(
            "split.parquet cambio despues del tuning ("
            f"{digest[:16]} != {selected['split_sha256'][:16]}): el modelo "
            "seleccionado no corresponde a estos datos. Re-ejecuta run_tuning.py"
        )

    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    params = selected["seleccion"]["params"]
    rounds = int(selected["seleccion"]["best_iteration"])

    print("[1/5] Cargando train y validation (test NO se carga) ...")
    splits = load_splits(split_file, names=EVAL_SETS)
    assert TEST_SET not in splits, "Este script no debe cargar test"
    train, valid = splits[TRAIN_SET], splits[VAL_SET]
    print(f"  train: n={train.n_rows} features={train.X.shape[1]}")
    print(f"  validation: n={valid.n_rows} features={valid.X.shape[1]}")

    print("[2/6] Verificando pliegues walk-forward internos de train ...")
    folds = walk_forward_folds(train.X["TransactionDT"])
    for c in check_folds(train.X["TransactionDT"], folds):
        print(f"  {'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    assert all(c["pass"] for c in check_folds(train.X["TransactionDT"], folds))

    print(
        f"[3/6] Corroboracion out-of-fold en train ({rounds} rondas, "
        f"objetivo: recall >= {TARGET_RECALL:.0%}) ..."
    )
    started = time.perf_counter()
    oof = out_of_fold_scores(params, rounds, train, folds, seed=SEED)
    for row in oof["folds"]:
        print(
            f"  pliegue {row['fold']}: n_train={row['n_train']} "
            f"n_scored={row['n_scored']} rondas={row['rounds']}"
        )
    print(f"  out-of-fold completo en {time.perf_counter() - started:.1f}s")
    print(
        f"  cobertura: {oof['n_scored']}/{train.n_rows} filas "
        f"({oof['coverage']:.1%}). Las {oof['n_unscored']} restantes son la "
        "ventana inicial que solo se uso para entrenar y no se puntua."
    )
    if oof["coverage"] < 0.25:
        raise ValueError(
            f"Cobertura out-of-fold insuficiente ({oof['coverage']:.1%}): la "
            "corroboracion seria sobre una muestra demasiado pequena"
        )
    mask = np.isfinite(oof["scores"])
    y_oof = np.asarray(train.y).astype(int)[mask]
    s_oof = oof["scores"][mask]
    oof_sweep = threshold_sweep(y_oof, s_oof)
    oof_sweep.to_csv(EVAL_REPORTS / OOF_SWEEP_CSV, index=False)
    del oof_sweep

    print("[4/6] Puntuando validation con el modelo final (el que se despliega) ...")
    model_file = ROOT / "artifacts" / "model_final.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Falta {model_file}. Ejecuta antes: run_tuning.py")
    estimator = joblib.load(model_file)
    y_valid = np.asarray(valid.y).astype(int)
    s_valid = np.asarray(estimator.predict_proba(valid.X))[:, 1]
    if int(y_valid.sum()) == int(y_oof.sum()):
        raise ValueError("validation y train puntuan el mismo numero de fraude")

    print(f"[5/6] Fijando el umbral en validation (recall >= {TARGET_RECALL:.0%}) ...")
    threshold = threshold_for_target_recall(y_valid, s_valid, TARGET_RECALL)
    valid_metrics = summarize(y_valid, s_valid, threshold=threshold)
    v_lo, v_hi = recall_interval(float(valid_metrics["recall"]), int(y_valid.sum()))
    print(
        f"  umbral operativo = {threshold:.6f}\n"
        f"  validation, base de calibracion (n={valid_metrics['n']}, "
        f"fraude={100 * valid_metrics['pos_rate']:.3f}%): "
        f"recall={valid_metrics['recall']:.4f} [{v_lo:.4f}, {v_hi:.4f}] "
        f"precision={valid_metrics['precision']:.4f} "
        f"predict_rate={valid_metrics['predict_rate']:.4f} "
        f"ROC-AUC={valid_metrics['roc_auc']:.4f} "
        f"PR-AUC={valid_metrics['pr_auc']:.4f}"
    )
    oof_metrics = summarize(y_oof, s_oof, threshold=threshold)
    o_lo, o_hi = recall_interval(float(oof_metrics["recall"]), int(y_oof.sum()))
    drift = float(oof_metrics["recall"]) - float(valid_metrics["recall"])
    print(
        f"  train out-of-fold, corroboracion (n={oof_metrics['n']}): "
        f"recall={oof_metrics['recall']:.4f} [{o_lo:.4f}, {o_hi:.4f}] "
        f"precision={oof_metrics['precision']:.4f} "
        f"predict_rate={oof_metrics['predict_rate']:.4f} "
        f"(deriva OOF - validation = {drift:+.4f})"
    )
    valid_sweep = threshold_sweep(y_valid, s_valid)
    valid_sweep.to_csv(EVAL_REPORTS / SWEEP_CSV, index=False)

    print("[6/6] Registrando el umbral ...")
    record: dict[str, Any] = {
        "criterio": f"recall objetivo >= {TARGET_RECALL}",
        "target_recall": TARGET_RECALL,
        "threshold": threshold,
        "origen_del_umbral": (
            "validation, unico bloque fuera de muestra puntuado por el modelo "
            "final; test no cargado"
        ),
        "config_id": selected["seleccion"]["config_id"],
        "params": params,
        "rounds": rounds,
        "seed": SEED,
        "split_sha256": digest,
        "test_usado": False,
        "oof_coverage": oof["coverage"],
        "oof_n_scored": oof["n_scored"],
        "oof_n_unscored": oof["n_unscored"],
        "oof_folds": oof["folds"],
        "calibracion_validation": valid_metrics,
        "calibracion_validation_recall_ci": [v_lo, v_hi],
        "corroboracion_oof_train": oof_metrics,
        "corroboracion_oof_train_recall_ci": [o_lo, o_hi],
        "deriva_recall_oof_menos_validation": drift,
    }
    path = EVAL_REPORTS / THRESHOLD_JSON
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  umbral -> {path}")
    print(f"  barrido validation -> {EVAL_REPORTS / SWEEP_CSV}")
    print(f"  barrido out-of-fold -> {EVAL_REPORTS / OOF_SWEEP_CSV}")
    print(
        "  Listo. El umbral queda fijado; run_evaluation.py puede consumir test "
        "una sola vez con este valor ya registrado."
    )


if __name__ == "__main__":
    main()
