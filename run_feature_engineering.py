from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import load_identity, load_transaction
from src.features.client import AGG_COLS, client_features, refined_uid
from src.features.columns import ADDR1, CARD1, DT, KEY
from src.features.features import compute_features

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = "train_processed.parquet"
FEATURES_FILE = "train_features.parquet"

IDENTITY_FILE = "train_identity.csv"


def _frame_digest(df: pd.DataFrame) -> str:
    values = df.select_dtypes(include="number").to_numpy()
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


def _client_coverage(txn: pd.DataFrame) -> None:
    """Cuanto de la matriz de cliente es valor real y cuanto es centinela.

    Una clave con poca cobertura produce centinelas: no rompe nada, pero su
    contribucion al modelo es nula. Sin este dato no se puede distinguir "el
    nivel refinado no aporta" de "el nivel refinado casi nunca aplica".

    Ojo con el alcance de la cifra: mide historial **dentro del propio frame**, no
    entre periodos. Cuantas filas de la ventana de test tienen historia del cliente
    en la ventana de entrenamiento es otra pregunta, y se mide en
    `docs/client_features_report.md` sobre `split.parquet`.
    """
    ordered = txn.sort_values(DT, kind="stable")
    client = client_features(ordered)
    uid = refined_uid(ordered)
    primary = ordered.groupby([CARD1, ADDR1], dropna=False).cumcount()
    print(
        f"  clave primaria {CARD1}+{ADDR1}: {float((primary > 0).mean()):.1%} de las "
        "filas tienen historial previo en el frame"
    )
    if uid is None:
        print("  clave refinada: no disponible (sin columna D1)")
        return
    refined_count = client["cnt_client_uid"]
    refined_means = client.filter(regex="_prev_uid$")
    con_historial = (refined_count > 0) | (refined_means.sum(axis=1) > 0)
    print(
        f"  clave refinada +D1n: {float(con_historial.mean()):.1%} de las filas tienen "
        f"historial previo | D1 nulo en {float(uid.isna().mean()):.1%}, esas filas "
        "degradan al nivel primario"
    )
    print(
        f"  columnas C agregadas: {len(AGG_COLS)} | M1-M9 excluidas (son "
        "categoricas en crudo)"
    )


def main() -> None:
    print("[1/5] Cargando datos raw ...")
    txn = load_transaction(DATA_DIR)
    identity_path = DATA_DIR / IDENTITY_FILE
    identity = load_identity(DATA_DIR) if identity_path.exists() else None
    print(f"  txn={txn.shape} identity={None if identity is None else identity.shape}")

    print("[2/5] Generando features (sin informacion futura) ...")
    features = compute_features(txn, identity)
    features = features.assign(**{KEY: txn[KEY].values})
    print(f"  features: {features.shape[1] - 1} columnas, {features.shape[0]} filas")
    print("  cobertura de las claves de cliente (T-E02):")
    _client_coverage(txn)

    print("[3/5] Verificando reproducibilidad (2 pasadas identicas) ...")
    features_2 = compute_features(txn, identity).assign(**{KEY: txn[KEY].values})
    digest_1 = _frame_digest(features)
    digest_2 = _frame_digest(features_2)
    identical = digest_1 == digest_2
    print(f"  digest 1: {digest_1[:16]}...")
    print(f"  digest 2: {digest_2[:16]}...")
    print(f"  IDENTICO: {identical}")

    print("[4/5] Combinando con features de preprocessing ...")
    processed_file = PROCESSED_DIR / PROCESSED_FILE
    processed = pd.read_parquet(processed_file)
    feature_cols = [c for c in features.columns if c != KEY]
    combined = processed.merge(features[[KEY, *feature_cols]], on=KEY, how="left")
    n_before = len(processed)
    n_after = len(combined)
    aligned = n_before == n_after and combined[feature_cols].notna().all().all()
    print(
        f"  filas: {n_before} -> {n_after} ; sin missing en features: {bool(aligned)}"
    )

    print("[5/5] Guardando resultado ...")
    for col in feature_cols:
        combined[col] = combined[col].astype("float32")
    out_file = PROCESSED_DIR / FEATURES_FILE
    combined.to_parquet(out_file, index=False)
    print(f"  guardado -> {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
