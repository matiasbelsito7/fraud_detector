from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from src.data.validation import load_transaction
from src.preprocessing.preprocessor import preprocess_dataset

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
PIPELINE_ARTIFACT = "preprocessor.joblib"
PROCESSED_FILE = "train_processed.parquet"


def _frame_digest(df) -> str:
    values = df.select_dtypes(include="number").to_numpy()
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


def main() -> None:
    print("[1/4] Cargando train_transaction.csv ...")
    txn = load_transaction(DATA_DIR)

    print("[2/4] Ajustando pipeline de preprocessing ...")
    roles, pipeline, processed = preprocess_dataset(txn)
    digest_1 = _frame_digest(processed)
    print(
        f"  roles: numeric={len(roles.numeric)} cat={len(roles.categorical)} "
        f"drop={roles.drop} passthrough={roles.passthrough}"
    )
    print(f"  shape procesado: {processed.shape}")

    print("[3/4] Verificando reproducibilidad (transform con pipeline guardado) ...")
    _roles, _pipeline, processed_2 = preprocess_dataset(txn, pipeline=pipeline)
    digest_2 = _frame_digest(processed_2)
    identical = digest_1 == digest_2
    print(f"  digest 1: {digest_1[:16]}...")
    print(f"  digest 2: {digest_2[:16]}...")
    print(f"  IDENTICO: {identical}")

    print("[4/4] Guardando artefactos ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    processed_file = PROCESSED_DIR / PROCESSED_FILE
    pipeline_file = ARTIFACTS_DIR / PIPELINE_ARTIFACT
    processed.to_parquet(processed_file, index=False)
    joblib.dump(pipeline, pipeline_file)
    print(
        f"  datos procesados -> {processed_file} "
        f"({processed_file.stat().st_size / 1e6:.1f} MB)"
    )
    print(f"  pipeline -> {pipeline_file}")


if __name__ == "__main__":
    main()
