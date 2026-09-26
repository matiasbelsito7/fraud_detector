from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.modeling.dataset import TRAIN_SET, VAL_SET, SplitData
from src.modeling.metrics import DEFAULT_THRESHOLD, summarize
from src.modeling.models import Candidate


@dataclass(frozen=True)
class RunResult:
    """Resultado de un candidato: metricas en validation y coste de ajuste."""

    name: str
    family: str
    metrics: dict[str, Any]
    fit_seconds: float
    n_train: int
    n_valid: int
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "modelo": self.name,
            "familia": self.family,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "fit_s": round(self.fit_seconds, 1),
            **self.metrics,
        }


def neg_pos_ratio(y: pd.Series) -> float:
    """Razon negativos/positivos del train; base de `scale_pos_weight`."""
    n_pos = int(np.asarray(y).sum())
    n_neg = int(len(y)) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("train requiere ambas clases para calcular el desbalance")
    return n_neg / n_pos


def subsample(
    X: pd.DataFrame, y: pd.Series, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Submuestra determinista por indice para acotar el uso de memoria."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[idx], y.iloc[idx]


def evaluate_candidate(
    candidate: Candidate,
    train: SplitData,
    valid: SplitData,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
) -> RunResult:
    """Ajusta en `train` y evalua en `validation`. `test` nunca se toca."""
    X_train, y_train = subsample(train.X, train.y, candidate.max_rows, seed=seed)
    started = time.perf_counter()
    candidate.estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    y_score = candidate.estimator.predict_proba(valid.X)[:, 1]
    metrics = summarize(valid.y, y_score, threshold=threshold)
    return RunResult(
        name=candidate.name,
        family=candidate.family,
        metrics=metrics,
        fit_seconds=fit_seconds,
        n_train=int(len(X_train)),
        n_valid=valid.n_rows,
        note=candidate.note,
    )


def run_candidates(
    candidates: list[Candidate],
    train: SplitData,
    valid: SplitData,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[RunResult]]:
    """Compara candidatos bajo condiciones equivalentes (misma division y features)."""
    results = [
        evaluate_candidate(c, train, valid, threshold=threshold, seed=seed)
        for c in candidates
    ]
    table = pd.DataFrame([r.as_row() for r in results])
    if not table.empty:
        table = table.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return table, results


def comparison_record(
    results: list[RunResult], seed: int, threshold: float
) -> dict[str, Any]:
    """Registro reproducible del experimento (specs §15, constitution §2 y §9)."""
    return {
        "seed": seed,
        "threshold": threshold,
        "entrenado_en": TRAIN_SET,
        "evaluado_en": VAL_SET,
        "test_usado": False,
        "modelos": [
            {
                "nombre": r.name,
                "familia": r.family,
                "nota": r.note,
                "n_train": r.n_train,
                "n_valid": r.n_valid,
                "fit_segundos": round(r.fit_seconds, 2),
                "metricas": r.metrics,
            }
            for r in results
        ],
    }
