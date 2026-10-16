"""Evalua el modelo final sobre `test` UNA vez, con el umbral ya fijado.

`test` solo se carga si se cumplen las precondiciones y el candado lo permite:
  0. el candado `config/test_consumed.json` no registra consumos previos
     (salvo `--force`, que deja traza).
  1. `reports/evaluation/threshold.json`, escrito por `run_threshold.py`.
  2. el digest de `split.parquet` sigue coincidiendo con el registrado.

El candado existe porque la comprobacion de digest dejo de proteger: al
re-ejecutarse `run_threshold.py` para T-E02 se regenero `threshold.json` con el
digest vigente, de modo que hoy esa precondicion pasaria. El registro de
consumos es independiente de los datos.

Este es el unico punto del pipeline donde se lee `test`. No se busca ningun
parametro ni ningun umbral aqui: se aplican los que ya estan registrados y se
miden. Por eso `load_splits` se llama con `FINAL_EVAL_SETS` (que incluye
`test`) en vez del default `EVAL_SETS` (que lo excluye): la excepcion es
visible en el codigo y no accidental.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
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
SENTINEL_FILE = ROOT / "config" / "test_consumed.json"
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


def load_consumptions(path: Path) -> list[dict[str, Any]]:
    """Consumos de `test` registrados en el candado (vacio si no existe)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("consumos", []))


def check_not_consumed(path: Path, force: bool) -> list[dict[str, Any]]:
    """Aborta si `test` ya figura consumido, salvo `force` explicito.

    Es independiente del digest de los datos: aunque `threshold.json` se
    regenere, el candado sigue bloqueando. Con `force` devuelve los consumos
    previos para que quede traza de que la reutilizacion fue deliberada.
    """
    consumos = load_consumptions(path)
    if consumos and not force:
        previos = ", ".join(
            str(c.get("fase") or c.get("fecha") or "?") for c in consumos
        )
        raise RuntimeError(
            f"test ya fue consumido ({previos}). No se vuelve a leer "
            "(constitution.md §5). Si de verdad corresponde, re-ejecuta con "
            "--force y quedara registrado en config/test_consumed.json."
        )
    return consumos


def register_consumption(path: Path, entrada: dict[str, Any]) -> None:
    """Anade un consumo al candado, preservando los anteriores."""
    consumos = load_consumptions(path)
    consumos.append(entrada)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"consumos": consumos}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def main(force: bool = False) -> None:
    split_file = PROCESSED_DIR / SPLIT_FILE
    if not split_file.exists():
        raise FileNotFoundError(
            f"Falta {split_file}. Ejecuta antes: run_preprocessing.py, "
            "run_feature_engineering.py, run_split.py"
        )

    print("[1/6] Comprobando el candado de test ...")
    consumos_previos = check_not_consumed(SENTINEL_FILE, force)
    if consumos_previos:
        print(
            f"  AVISO: test ya figura consumido ({len(consumos_previos)} "
            "consumo/s previo/s); se continua por --force explicito"
        )
    else:
        print("  candado limpio: test no registra consumos previos")

    print("[2/6] Comprobando precondiciones (test sigue sin cargarse) ...")
    record = _check_preconditions(split_file)
    threshold = float(record["threshold"])
    assert TEST_SET not in EVAL_SETS, "el default de load_splits debe excluir test"
    print(f"  umbral ya fijado: {threshold:.6f} (config {record['config_id']})")
    print(f"  origen: {record['origen_del_umbral']}")

    print("[3/6] Cargando test por primera y unica vez ...")
    splits = load_splits(split_file, names=FINAL_EVAL_SETS)
    assert TEST_SET in splits
    test = splits[TEST_SET]
    print(f"  test: n={test.n_rows} features={test.X.shape[1]}")
    print("  (fin del uso de test: no se vuelve a leer en este pipeline)")

    print("[4/6] Puntuando con el modelo final y el umbral registrado ...")
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

    print("[5/6] Guardando metricas, barrido y curvas de test ...")
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

    register_consumption(
        SENTINEL_FILE,
        {
            "fecha": date.today().isoformat(),
            "digest_split": record["split_sha256"],
            "config_id": record["config_id"],
            "rounds": record["rounds"],
            "reporte": path.relative_to(ROOT).as_posix(),
            "forzado": bool(consumos_previos),
        },
    )
    print(f"  candado actualizado -> {SENTINEL_FILE}")

    print("[6/6] Resumen de la evaluacion final ...")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Permite re-leer test aunque ya figure consumido. La reutilizacion "
            "queda registrada en config/test_consumed.json."
        ),
    )
    args = parser.parse_args()
    main(force=args.force)
