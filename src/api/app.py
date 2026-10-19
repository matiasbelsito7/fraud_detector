from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import HealthResponse, PredictionResponse, PredictRequest
from src.inference.pipeline import InferencePipeline

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "deployment.json"
CONFIG_ENV = "FRAUD_DEPLOYMENT_CONFIG"


def create_app(
    pipeline: InferencePipeline | None = None,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    verify_digest: bool = True,
) -> FastAPI:
    """Crea la app; el modelo se carga una vez en el arranque (lifespan)."""
    if config_path is None:
        config_path = os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = (
            pipeline
            if pipeline is not None
            else InferencePipeline.load(
                config_path, root=root, verify_digest=verify_digest
            )
        )
        yield

    app = FastAPI(
        title="Fraud Detector API",
        version="0.1.0",
        description="Inferencia de fraude sobre el pipeline de la Fase L.",
        lifespan=lifespan,
    )

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
        try:
            prediction = loaded.predict(
                payload.transaction,
                history=_records_frame(payload.history),
                identity=_record_frame(payload.identity),
                history_identity=_records_frame(payload.history_identity),
                threshold=payload.threshold,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
