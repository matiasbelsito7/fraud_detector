from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import load_identity, load_transaction
from src.features.features import KEY, compute_features

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = "train_processed.parquet"
FEATURES_FILE = "train_features.parquet"

IDENTITY_FILE = "train_identity.csv"


def _frame_digest(df: pd.DataFrame) -> str:
    values = df.select_dtypes(include="number").to_numpy()
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


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
