from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.features.columns import KEY
from src.features.features import compute_features
from src.inference.pipeline import InferencePipeline, file_digest
from src.preprocessing.preprocessor import preprocess_dataset


def _toy_raw() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 20
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1000, 1000 + n),
            "TransactionDT": np.arange(n) * 3600 + 86400,
            "TransactionAmt": rng.uniform(10, 500, n).round(2),
            "ProductCD": rng.choice(["W", "C"], n),
            "card1": rng.integers(1, 4, n),
            "card2": rng.integers(100, 103, n),
            "card3": 150.0,
            "card4": "visa",
            "card6": "debit",
            "addr1": rng.integers(100, 103, n),
            "addr2": 87.0,
            "D1": rng.integers(50, 80, n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "R_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "C1": rng.integers(0, 5, n),
            "C2": rng.integers(0, 5, n),
            "C3": rng.integers(0, 5, n),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "M1": "T",
            "isFraud": rng.integers(0, 2, n),
        }
    )


def _toy_identity(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            KEY: raw[KEY].iloc[::2],
            "DeviceType": "mobile",
        }
    )


@pytest.fixture()
def toy(tmp_path: Path) -> dict[str, Any]:
    raw = _toy_raw()
    identity = _toy_identity(raw)
    _roles, preprocessor, processed = preprocess_dataset(raw)
    features = compute_features(raw, identity)
    feature_cols = [c for c in features.columns if c != KEY]
    X = pd.concat(
        [
            processed.drop(columns=[KEY, "isFraud"]),
            features[feature_cols],
        ],
        axis=1,
    )
    y = raw["isFraud"]
    model = LGBMClassifier(
        n_estimators=15,
        num_leaves=4,
        min_child_samples=2,
        random_state=42,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pre_path = artifacts / "preprocessor.joblib"
    model_path = artifacts / "model.joblib"
    joblib.dump(preprocessor, pre_path)
    joblib.dump(model, model_path)

    deployment_path = tmp_path / "config" / "deployment.json"
    deployment_path.parent.mkdir()
    deployment_path.write_text(
        json.dumps(
            {
                "variant": "toy",
                "model_artifact": "artifacts/model.joblib",
                "model_sha256": file_digest(model_path),
                "preprocessor_artifact": "artifacts/preprocessor.joblib",
                "preprocessor_sha256": file_digest(pre_path),
                "threshold": 0.5,
                "threshold_source": "toy",
                "config_id": 1,
                "rounds": 15,
                "split_sha256": "a" * 64,
                "target_recall": 0.8,
                "test_usado": False,
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": tmp_path,
        "raw": raw,
        "identity": identity,
        "X": X,
        "model": model,
        "deployment": deployment_path,
        "pipeline": InferencePipeline.load(deployment_path, root=tmp_path),
    }
