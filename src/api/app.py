from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.schemas import HealthResponse, PredictionResponse, PredictRequest
from src.inference.pipeline import InferencePipeline
from src.monitoring.drift import DriftMonitor

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "deployment.json"
CONFIG_ENV = "FRAUD_DEPLOYMENT_CONFIG"

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Solicitudes HTTP recibidas.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Latencia de las solicitudes HTTP.",
    ["method", "route"],
)
PREDICTION_COUNT = Counter(
    "fraud_api_predictions_total",
    "Predicciones servidas por etiqueta.",
    ["label"],
)
DRIFT_PSI = Gauge(
    "fraud_drift_psi",
    "PSI de la feature contra la referencia de train.",
    ["feature"],
)
DRIFT_OBSERVATIONS = Gauge(
    "fraud_drift_observations",
    "Observaciones retenidas por el monitor de drift.",
)
DRIFT_DRIFTED = Gauge(
    "fraud_drift_features_drifted",
    "Features con drift significativo (PSI >= 0.25).",
)
DRIFT_MAX_PSI = Gauge(
    "fraud_drift_max_psi",
    "PSI maximo observado entre las features monitorizadas.",
)


def create_app(
    pipeline: InferencePipeline | None = None,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    verify_digest: bool = True,
    drift_monitor: DriftMonitor | None = None,
) -> FastAPI:
    """Crea la app; el modelo se carga una vez en el arranque (lifespan)."""
    if config_path is None:
        config_path = os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = (
            pipeline
            if pipeline is not None
            else InferencePipeline.load(
                config_path, root=root, verify_digest=verify_digest
            )
        )
        app.state.pipeline = loaded
        if drift_monitor is not None:
            app.state.drift = drift_monitor
        elif loaded.deployment.drift_reference is not None:
            reference = loaded.deployment.drift_reference
            if not reference.exists():
                raise FileNotFoundError(
                    f"Falta la referencia de drift configurada: {reference}"
                )
            app.state.drift = DriftMonitor.from_file(reference)
        else:
            app.state.drift = None
        yield

    app = FastAPI(
        title="Fraud Detector API",
        version="0.1.0",
        description="Inferencia de fraude sobre el pipeline de la Fase L.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def monitor_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            REQUEST_COUNT.labels(request.method, path, str(status)).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(
                time.perf_counter() - started
            )

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        monitor = getattr(request.app.state, "drift", None)
        if monitor is not None:
            snapshot = monitor.snapshot()
            DRIFT_OBSERVATIONS.set(snapshot["n_observations"])
            DRIFT_DRIFTED.set(
                sum(
                    1
                    for item in snapshot["features"]
                    if item["status"] == "significant"
                )
            )
            if snapshot["max_psi"] is not None:
                DRIFT_MAX_PSI.set(snapshot["max_psi"])
            for item in snapshot["features"]:
                DRIFT_PSI.labels(item["feature"]).set(item["psi"] or 0.0)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/drift")
    def drift(request: Request) -> dict[str, Any]:
        monitor = getattr(request.app.state, "drift", None)
        if monitor is None:
            return {
                "enabled": False,
                "reason": "sin referencia de drift configurada",
            }
        return {"enabled": True, **monitor.snapshot()}

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded: InferencePipeline = request.app.state.pipeline
        deployment = loaded.deployment
        return HealthResponse(
            variant=deployment.variant,
            config_id=deployment.config_id,
            threshold=deployment.threshold,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
        loaded: InferencePipeline = request.app.state.pipeline
        monitor: DriftMonitor | None = getattr(request.app.state, "drift", None)
        try:
            prediction = loaded.predict(
                payload.transaction,
                history=_records_frame(payload.history),
                identity=_record_frame(payload.identity),
                history_identity=_records_frame(payload.history_identity),
                threshold=payload.threshold,
                observe=monitor.observe if monitor is not None else None,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        PREDICTION_COUNT.labels(prediction.label).inc()
        return PredictionResponse(**prediction.as_dict())

    return app


def _records_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame | None:
    if not records:
        return None
    return pd.DataFrame(list(records))


def _record_frame(record: dict[str, Any] | None) -> pd.DataFrame | None:
    if record is None:
        return None
    return pd.DataFrame([record])
