"""Evalua el modelo final sobre `test` UNA vez, con el umbral ya fijado.

`test` solo se carga si existen las dos precondiciones:
  1. `reports/evaluation/threshold.json`, escrito por `run_threshold.py`.
  2. el digest de `split.parquet` sigue coincidiendo con el de la Fase H.

Este es el unico punto del pipeline donde se lee `test`. No se busca ningun
parametro ni ningun umbral aqui: se aplican los que ya estan registrados y se
miden. Por eso `load_splits` se llama con `FINAL_EVAL_SETS` (que incluye
`test`) en vez del default `EVAL_SETS` (que lo excluye): la excepcion es
visible en el codigo y no accidental.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.modeling.dataset import EVAL_SETS, TEST_SET, TRAIN_SET, VAL_SET, load_splits
from src.modeling.evaluate import (
    TARGET_RECALL,
    curve_points,
    recall_interval,
    threshold_sweep,
)
from src.modeling.metrics import summarize

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
TUNING_DIR = REPORTS_DIR / "tuning"
EVAL_REPORTS = REPORTS_DIR / "evaluation"

SPLIT_FILE = "split.parquet"
MODEL_FILE = ROOT / "artifacts" / "model_final.joblib"
SELECTED_JSON = TUNING_DIR / "selected.json"
THRESHOLD_JSON = EVAL_REPORTS / "threshold.json"
METRICS_JSON = "test_metrics.json"
SWEEP_CSV = "test_sweep.csv"
ROC_CURVE_CSV = "test_roc_curve.csv"
PR_CURVE_CSV = "test_pr_curve.csv"

# Unicauple que incluye test. El default de load_splits lo excluye a proposito.
FINAL_EVAL_SETS = (TRAIN_SET, VAL_SET, TEST_SET)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_preconditions(split_file: Path) -> dict[str, Any]:
    if not THRESHOLD_JSON.exists():
        raise FileNotFoundError(
            f"Falta {THRESHOLD_JSON}. El umbral debe fijarse antes de tocar "
            "test; ejecuta antes: run_threshold.py"
        )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Falta {MODEL_FILE}. Ejecuta antes: run_tuning.py")
    if not SELECTED_JSON.exists():
        raise FileNotFoundError(f"Falta {SELECTED_JSON}. Ejecuta antes: run_tuning.py")

    record = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    if record.get("test_usado") is not False:
        raise ValueError("threshold.json no certifica test_usado=false")
    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    if selected.get("test_usado") is not False:
        raise ValueError("selected.json no certifica test_usado=false")
    if record["config_id"] != selected["seleccion"]["config_id"]:
        raise ValueError(
            f"El umbral se fijo para la config {record['config_id']} pero la "
            f"seleccionada es {selected['seleccion']['config_id']}"
        )
    digest = _file_digest(split_file)
    if digest != record["split_sha256"]:
        raise ValueError(
            f"split.parquet cambio despues de fijarse el umbral "
            f"({digest[:16]} != {record['split_sha256'][:16]}): las metricas "
            "no serian comparables. Re-ejecuta run_tuning.py y run_threshold.py"
        )
    return record


def main() -> None:
    split_file = PROCESSED_DIR / SPLIT_FILE
    if not split_file.exists():
        raise FileNotFoundError(
            f"Falta {split_file}. Ejecuta antes: run_preprocessing.py, "
            "run_feature_engineering.py, run_split.py"
        )

    print("[1/5] Comprobando precondiciones (test sigue sin cargarse) ...")
    record = _check_preconditions(split_file)
    threshold = float(record["threshold"])
    assert TEST_SET not in EVAL_SETS, "el default de load_splits debe excluir test"
    print(f"  umbral ya fijado: {threshold:.6f} (config {record['config_id']})")
    print(f"  origen: {record['origen_del_umbral']}")

    print("[2/5] Cargando test por primera y unica vez ...")
    splits = load_splits(split_file, names=FINAL_EVAL_SETS)
    assert TEST_SET in splits
    test = splits[TEST_SET]
    print(f"  test: n={test.n_rows} features={test.X.shape[1]}")
    print("  (fin del uso de test: no se vuelve a leer en este pipeline)")

    print("[3/5] Puntuando con el modelo final y el umbral registrado ...")
    estimator = joblib.load(MODEL_FILE)
    y_test = np.asarray(test.y).astype(int)
    s_test = np.asarray(estimator.predict_proba(test.X))[:, 1]
    metrics: dict[str, Any] = summarize(y_test, s_test, threshold=threshold)
    lo, hi = recall_interval(float(metrics["recall"]), int(y_test.sum()))
    metrics["recall_ci_95"] = [lo, hi]
    metrics["cumple_objetivo_recall"] = bool(metrics["recall"] >= TARGET_RECALL)
    print(
        f"  test (n={metrics['n']}, fraude={100 * metrics['pos_rate']:.3f}%): "
        f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}\n"
        f"  umbral={threshold:.6f} -> recall={metrics['recall']:.4f} "
        f"[{lo:.4f}, {hi:.4f}] precision={metrics['precision']:.4f} "
        f"f1={metrics['f1']:.4f} alertas={100 * metrics['predict_rate']:.2f}%\n"
        f"  matriz: tp={metrics['tp']} fp={metrics['fp']} "
        f"fn={metrics['fn']} tn={metrics['tn']}\n"
        f"  objetivo recall >= {TARGET_RECALL:.0%}: "
        f"{'CUMPLE' if metrics['cumple_objetivo_recall'] else 'NO CUMPLE'}"
    )

    print("[4/5] Guardando metricas, barrido y curvas de test ...")
    EVAL_REPORTS.mkdir(parents=True, exist_ok=True)
    threshold_sweep(y_test, s_test).to_csv(EVAL_REPORTS / SWEEP_CSV, index=False)
    curves = curve_points(y_test, s_test)
    # ROC y PR no comparten numero de puntos: van en archivos separados
    pd.DataFrame(
        {
            "fpr": curves["fpr"],
            "tpr": curves["tpr"],
            "threshold": curves["roc_threshold"],
        }
    ).to_csv(EVAL_REPORTS / ROC_CURVE_CSV, index=False)
    pd.DataFrame(
        {
            "recall": curves["recall"],
            "precision": curves["precision"],
            "threshold": curves["pr_threshold"],
        }
    ).to_csv(EVAL_REPORTS / PR_CURVE_CSV, index=False)
    out: dict[str, Any] = {
        "conjunto": TEST_SET,
        "umbral": threshold,
        "criterio": f"recall objetivo >= {TARGET_RECALL}",
        "config_id": record["config_id"],
        "rounds": record["rounds"],
        "split_sha256": record["split_sha256"],
        "origen_del_umbral": record["origen_del_umbral"],
        "umbral_fijado_antes_de_test": True,
        "metricas_test": metrics,
        "referencia_validacion": record["calibracion_validation"],
        "referencia_oof_train": record["corroboracion_oof_train"],
    }
    path = EVAL_REPORTS / METRICS_JSON
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metricas -> {path}")
    print(f"  barrido -> {EVAL_REPORTS / SWEEP_CSV}")
    print(f"  curva ROC -> {EVAL_REPORTS / ROC_CURVE_CSV}")
    print(f"  curva PR -> {EVAL_REPORTS / PR_CURVE_CSV}")

    print("[5/5] Resumen de la evaluacion final ...")
    ref = record["calibracion_validation"]
    print(
        f"  {'bloque':<12} {'n':>8} {'ROC-AUC':>9} {'PR-AUC':>8} "
        f"{'recall':>7} {'prec':>7} {'alertas':>8}"
    )
    for name, m in (
        ("train OOF", record["corroboracion_oof_train"]),
        ("validation", ref),
        ("test", metrics),
    ):
        print(
            f"  {name:<12} {int(m['n']):>8} {m['roc_auc']:>9.4f} "
            f"{m['pr_auc']:>8.4f} {m['recall']:>7.4f} "
            f"{m['precision']:>7.4f} {100 * m['predict_rate']:>7.2f}%"
        )
    print("\n  test ya fue consumido. No se volvera a usar para elegir nada.")


if __name__ == "__main__":
    main()
