from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.monitoring.drift import DriftMonitor, build_reference


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _minimal_transaction() -> dict[str, Any]:
    return {
        "TransactionDT": 86400 * 3,
        "TransactionAmt": 100.0,
        "card1": 1,
    }


@pytest.fixture()
def client(toy: dict[str, Any]) -> Iterator[TestClient]:
    with TestClient(create_app(pipeline=toy["pipeline"])) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "variant": "toy",
        "config_id": 1,
        "threshold": 0.5,
    }


def test_metrics(client: TestClient) -> None:
    prediction = client.post("/predict", json={"transaction": _minimal_transaction()})
    assert prediction.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "fraud_api_requests_total" in text
    assert "fraud_api_request_duration_seconds" in text
    assert "fraud_api_predictions_total" in text


def test_predict_con_historial(client: TestClient, toy: dict[str, Any]) -> None:
    raw: pd.DataFrame = toy["raw"]
    payload = {
        "transaction": _records(raw.iloc[[-1]])[0],
        "history": _records(raw.iloc[:-1]),
        "identity": _records(toy["identity"].tail(1))[0],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    direct = toy["pipeline"].predict(
        raw.iloc[-1],
        history=raw.iloc[:-1],
        identity=toy["identity"].tail(1),
    )
    assert body["probability"] == pytest.approx(direct.probability, abs=1e-12)
    assert body["threshold"] == pytest.approx(0.5)
    assert body["variant"] == "toy"
    assert body["config_id"] == 1
    assert body["n_history"] == len(raw) - 1
    assert body["label"] in {"fraude", "legitima"}
    assert body["is_fraud"] == (body["probability"] >= 0.5)


def test_predict_sin_historial(client: TestClient) -> None:
    response = client.post("/predict", json={"transaction": _minimal_transaction()})
    assert response.status_code == 200, response.text
    assert response.json()["n_history"] == 0


def test_predict_historial_futuro_400(client: TestClient, toy: dict[str, Any]) -> None:
    raw: pd.DataFrame = toy["raw"]
    payload = {
        "transaction": _records(raw.iloc[[5]])[0],
        "history": _records(raw.iloc[6:]),
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "leakage temporal" in response.json()["detail"]


def test_predict_columnas_requeridas_422(client: TestClient) -> None:
    response = client.post("/predict", json={"transaction": {"TransactionAmt": 10}})
    assert response.status_code == 422


def test_predict_umbral_fuera_de_rango_422(client: TestClient) -> None:
    payload = {"transaction": _minimal_transaction(), "threshold": 1.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_umbral_configurable(client: TestClient) -> None:
    payload = {"transaction": _minimal_transaction(), "threshold": 0.000001}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["is_fraud"] is True


def test_identity_sin_transaction_id_400(client: TestClient) -> None:
    payload = {
        "transaction": _minimal_transaction(),
        "identity": {"DeviceType": "mobile"},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "TransactionID" in response.json()["detail"]


def test_drift_sin_referencia(client: TestClient) -> None:
    response = client.get("/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert "sin referencia" in body["reason"]


def test_drift_observa_predicciones(toy: dict[str, Any]) -> None:
    features = list(toy["X"].columns[:2])
    reference = build_reference(toy["X"], features, bins=4)
    monitor = DriftMonitor(reference)
    with TestClient(
        create_app(pipeline=toy["pipeline"], drift_monitor=monitor)
    ) as test_client:
        before = test_client.get("/drift").json()
        assert before["enabled"] is True
        assert before["n_observations"] == 0
        assert before["status"] == "no_data"

        prediction = test_client.post(
            "/predict", json={"transaction": _minimal_transaction()}
        )
        assert prediction.status_code == 200, prediction.text

        after = test_client.get("/drift").json()
        assert after["enabled"] is True
        assert after["n_observations"] == 1
        assert {item["feature"] for item in after["features"]} == set(features)
        assert all(item["psi"] is not None for item in after["features"])

        metrics = test_client.get("/metrics")
        assert metrics.status_code == 200
        assert "fraud_drift_psi" in metrics.text
        assert "fraud_drift_observations 1.0" in metrics.text
        assert "fraud_drift_features_drifted" in metrics.text


def test_drift_se_carga_desde_deployment(toy: dict[str, Any]) -> None:
    reference = build_reference(
        toy["X"],
        list(toy["X"].columns[:2]),
        bins=4,
        metadata={"source": "train"},
    )
    reference.save(toy["root"] / "artifacts" / "drift.json")
    data = json.loads(toy["deployment"].read_text(encoding="utf-8"))
    data["drift_reference"] = "artifacts/drift.json"
    toy["deployment"].write_text(json.dumps(data), encoding="utf-8")

    with TestClient(
        create_app(config_path=toy["deployment"], root=toy["root"])
    ) as test_client:
        body = test_client.get("/drift").json()
        assert body["enabled"] is True
        assert body["reference"]["source"] == "train"
        assert len(body["features"]) == 2
