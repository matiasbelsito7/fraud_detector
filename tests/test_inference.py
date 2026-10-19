from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.inference.pipeline import Deployment, InferencePipeline

ROOT = Path(__file__).resolve().parent.parent


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
    probabilities = model.predict_proba(X.iloc[[-1]])
    batch = float(probabilities[0][1])
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
