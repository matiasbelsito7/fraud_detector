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
from src.inference.pipeline import Deployment, InferencePipeline, file_digest
from src.preprocessing.preprocessor import preprocess_dataset

ROOT = Path(__file__).resolve().parent.parent


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


def test_inferencia_fila_unica_igual_a_lote(toy: dict[str, Any]) -> None:
    raw = toy["raw"]
    pipeline: InferencePipeline = toy["pipeline"]
    model: LGBMClassifier = toy["model"]
    X: pd.DataFrame = toy["X"]

    prediction = pipeline.predict(
        raw.iloc[-1],
        history=raw.iloc[:-1],
        identity=toy["identity"].tail(1),
    )
    batch = float(model.predict_proba(X.iloc[[-1]])[0, 1])
    assert prediction.probability == pytest.approx(batch, abs=1e-6)
    assert prediction.is_fraud == (batch >= 0.5)
    assert prediction.threshold == pytest.approx(0.5)
    assert prediction.n_history == len(raw) - 1
    assert prediction.variant == "toy"


def test_historial_futuro_rechazado(toy: dict[str, Any]) -> None:
    raw = toy["raw"]
    pipeline: InferencePipeline = toy["pipeline"]
    target = raw.iloc[5]
    future = raw.iloc[6:]
    with pytest.raises(ValueError, match="leakage temporal"):
        pipeline.predict(target, history=future)


def test_columnas_faltantes_se_rellenan(toy: dict[str, Any]) -> None:
    pipeline: InferencePipeline = toy["pipeline"]
    minimal = {
        "TransactionDT": 86400 * 3,
        "TransactionAmt": 100.0,
        "card1": 1,
    }
    prediction = pipeline.predict(minimal)
    assert np.isfinite(prediction.probability)
    assert prediction.n_history == 0
    assert prediction.probability == pytest.approx(
        pipeline.predict(minimal).probability
    )


def test_umbral_configurable(toy: dict[str, Any]) -> None:
    pipeline: InferencePipeline = toy["pipeline"]
    minimal = {
        "TransactionDT": 86400 * 3,
        "TransactionAmt": 100.0,
        "card1": 1,
    }
    assert pipeline.predict(minimal, threshold=0.000001).is_fraud
    assert not pipeline.predict(minimal, threshold=0.999999).is_fraud


def test_digest_invalido_aborta(toy: dict[str, Any]) -> None:
    path = toy["deployment"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="no coincide con el sha256"):
        InferencePipeline.load(path, root=toy["root"])


def test_deployment_valida_umbral(toy: dict[str, Any]) -> None:
    path = toy["deployment"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["threshold"] = 1.5
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Umbral fuera"):
        Deployment.load(path, root=toy["root"], verify_digest=False)


def test_no_lee_test_ni_split() -> None:
    source = (ROOT / "src" / "inference").glob("*.py")
    text = "\n".join(path.read_text(encoding="utf-8") for path in source)
    assert "load_splits" not in text
    assert "test_transaction.csv" not in text
    assert "split.parquet" not in text
