from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUIRED_TRANSACTION_COLUMNS = ("TransactionDT", "TransactionAmt", "card1")


class PredictRequest(BaseModel):
    """Una transaccion raw, con historial e identidad opcionales."""

    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)
    identity: dict[str, Any] | None = None
    history_identity: list[dict[str, Any]] = Field(default_factory=list)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_transaction(self) -> PredictRequest:
        missing = [
            column
            for column in REQUIRED_TRANSACTION_COLUMNS
            if column not in self.transaction or _is_null(self.transaction[column])
        ]
        if missing:
            raise ValueError(f"transaction sin columnas requeridas: {missing}")
        return self


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    is_fraud: bool
    label: Literal["fraude", "legitima"]
    threshold: float
    variant: str
    config_id: int
    n_history: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    variant: str
    config_id: int
    threshold: float


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
