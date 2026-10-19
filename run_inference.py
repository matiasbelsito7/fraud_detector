"""Prueba end-to-end del pipeline de inferencia (T-L01).

Toma una transaccion real de `validation`, le pasa como historial solo las filas
anteriores de `train_transaction.csv` y predice con el artefacto desplegable. No
calcula metricas: solo comprueba que una transaccion atraviesa preprocessing,
feature engineering, modelo y umbral, y que la probabilidad coincide con la que
el pipeline por lotes produjo para esa misma fila.

`test` no se toca: la transaccion se exige dentro de la ventana de `validation`
y el historial se corta antes de ella. La lectura filtrada de `split.parquet`
solo recupera esa fila para comparar.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.validation import IDENTITY_FILE, load_identity, load_transaction
from src.features.columns import AMT, CARD1, DT, KEY
from src.inference.pipeline import InferencePipeline
from src.modeling.dataset import feature_columns
from src.split.split import TRAIN_WEEKS, VAL_WEEKS, week_index

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_FILE = "split.parquet"
REPORTS_DIR = ROOT / "reports" / "inference"
DEPLOYMENT_FILE = ROOT / "config" / "deployment.json"


def _pick_target(
    txn: pd.DataFrame, transaction_id: int | None
) -> tuple[int, pd.Series]:
    """Posicion y fila de una transaccion de validacion con la que demostrar."""
    if transaction_id is not None:
        matches = txn.index[txn[KEY] == transaction_id]
        if len(matches) != 1:
            raise ValueError(f"TransactionID {transaction_id} no encontrado")
        position = int(matches[0])
    else:
        weeks = week_index(txn[DT])
        valid = (weeks >= TRAIN_WEEKS) & (weeks < TRAIN_WEEKS + VAL_WEEKS)
        usable = valid & txn[CARD1].notna() & txn[AMT].notna()
        candidates = txn.index[usable]
        if len(candidates) == 0:
            raise ValueError("No hay transacciones de validation utilizables")
        position = int(candidates[0])

    row = txn.loc[position]
    week = int(week_index(txn[DT]).loc[position])
    if not TRAIN_WEEKS <= week < TRAIN_WEEKS + VAL_WEEKS:
        raise ValueError(
            f"La transaccion {row[KEY]} cae en la semana {week}, fuera de "
            f"validation ({TRAIN_WEEKS}-{TRAIN_WEEKS + VAL_WEEKS - 1}). "
            "No se usa test en esta demostracion."
        )
    return position, row


def _batch_probability(pipeline: InferencePipeline, transaction_id: int) -> float:
    """Probabilidad que el pipeline por lotes asigno a la misma fila."""
    path = PROCESSED_DIR / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")
    row = pd.read_parquet(path, filters=[(KEY, "==", int(transaction_id))])
    if len(row) != 1:
        raise ValueError(f"Se esperaba una fila para {transaction_id}, hay {len(row)}")
    if str(row["split_set"].iloc[0]) != "validation":
        raise ValueError(
            f"La fila {transaction_id} pertenece a {row['split_set'].iloc[0]}, "
            "no a validation"
        )
    X = row[feature_columns(row)].astype("float32")
    if list(X.columns) != pipeline.feature_order:
        raise ValueError("Las columnas del split no coinciden con el modelo")
    return float(pipeline.model.predict_proba(X[pipeline.feature_order])[0, 1])


def main(transaction_id: int | None = None) -> int:
    print("[1/5] Cargando datos raw e identidad ...")
    txn = load_transaction(DATA_DIR)
    identity_path = DATA_DIR / IDENTITY_FILE
    identity = load_identity(DATA_DIR) if identity_path.exists() else None
    print(
        f"  transacciones={len(txn)} identity={None if identity is None else len(identity)}"
    )

    print("[2/5] Seleccionando una transaccion de validation ...")
    position, target = _pick_target(txn, transaction_id)
    target_id = int(target[KEY])
    target_dt = int(target[DT])
    history = txn.iloc[:position]
    if int(history[DT].max()) > target_dt:
        raise AssertionError("El historial contiene informacion futura")
    target_identity = None
    if identity is not None:
        target_identity = identity[identity[KEY] == target_id]
    print(
        f"  TransactionID={target_id} DT={target_dt} "
        f"historial={len(history)} filas "
        f"identity={'si' if target_identity is not None else 'no'}"
    )

    print("[3/5] Cargando pipeline desplegable ...")
    pipeline = InferencePipeline.load(DEPLOYMENT_FILE, root=ROOT)
    print(
        f"  variante={pipeline.deployment.variant} "
        f"config={pipeline.deployment.config_id} "
        f"rondas={pipeline.deployment.rounds} "
        f"umbral={pipeline.deployment.threshold:.6f} "
        f"features={len(pipeline.feature_order)}"
    )

    print("[4/5] Prediciendo una transaccion ...")
    prediction = pipeline.predict(
        target.to_dict(),
        history=history,
        identity=target_identity,
    )
    batch_probability = _batch_probability(pipeline, target_id)
    difference = abs(prediction.probability - batch_probability)
    print(
        f"  probabilidad={prediction.probability:.6f} "
        f"clase={prediction.label} "
        f"umbral={prediction.threshold:.6f}"
    )
    print(f"  pipeline por lotes={batch_probability:.6f} diferencia={difference:.2e}")
    if difference > 1e-6:
        raise AssertionError(
            f"La inferencia no reproduce el pipeline por lotes: {difference:.2e} > 1e-6"
        )
    print("  consistencia verificada (diferencia <= 1e-6)")

    print("[5/5] Guardando evidencia ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "fecha": datetime.now(UTC).isoformat(),
        "test_usado": False,
        "conjunto": "validation",
        "transaction_id": target_id,
        "transaction_dt": target_dt,
        "historial_filas": int(len(history)),
        "identity_disponible": target_identity is not None,
        "modelo": {
            "variante": pipeline.deployment.variant,
            "config_id": pipeline.deployment.config_id,
            "rondas": pipeline.deployment.rounds,
            "n_features": len(pipeline.feature_order),
            "model_sha256": pipeline.deployment.model_sha256,
            "preprocessor_sha256": pipeline.deployment.preprocessor_sha256,
            "split_sha256": pipeline.deployment.split_sha256,
        },
        "prediccion": prediction.as_dict(),
        "pipeline_por_lotes": batch_probability,
        "diferencia_abs": difference,
        "consistente": bool(difference <= 1e-6),
    }
    path = REPORTS_DIR / "example.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  evidencia -> {path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--transaction-id",
        type=int,
        default=None,
        help="TransactionID de validation a puntuar (por defecto, la primera).",
    )
    args = parser.parse_args()
    raise SystemExit(main(transaction_id=args.transaction_id))
