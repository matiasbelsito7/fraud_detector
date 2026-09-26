from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5

PRIMARY_METRIC = "pr_auc"
SECONDARY_METRIC = "roc_auc"


def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true_a = np.asarray(y_true).astype(int)
    y_score_a = np.asarray(y_score, dtype=float).ravel()
    if y_true_a.shape != y_score_a.shape:
        raise ValueError(
            f"y_true e y_score con distinta forma: {y_true_a.shape} vs {y_score_a.shape}"
        )
    if np.unique(y_true_a).size < 2:
        raise ValueError("Se requieren ambas clases para calcular ROC-AUC y PR-AUC")
    return y_true_a, y_score_a


def roc_auc(y_true: Any, y_score: Any) -> float:
    y_true_a, y_score_a = _as_arrays(y_true, y_score)
    return float(roc_auc_score(y_true_a, y_score_a))


def pr_auc(y_true: Any, y_score: Any) -> float:
    """PR-AUC como precision-recall promedio (adecuada para clases desbalanceadas)."""
    y_true_a, y_score_a = _as_arrays(y_true, y_score)
    return float(average_precision_score(y_true_a, y_score_a))


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    y_true_a = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred).astype(int)
    if y_true_a.shape != y_pred_a.shape:
        raise ValueError("y_true e y_pred con distinta forma")
    tn, fp, fn, tp = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(
    y_true: Any, y_score: Any, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, float | int]:
    """Conjunto minimo de metricas de specs.md 8.1 sobre un threshold dado.

    La eleccion del threshold operativo corresponde a la Fase I (specs §8.2);
    aqui se reportan las metricas en un threshold de referencia.
    """
    y_true_a, y_score_a = _as_arrays(y_true, y_score)
    y_pred = (y_score_a >= threshold).astype(int)
    counts = confusion_counts(y_true_a, y_pred)
    return {
        "n": int(y_true_a.size),
        "pos_rate": float(y_true_a.mean()),
        "threshold": float(threshold),
        "roc_auc": roc_auc(y_true_a, y_score_a),
        "pr_auc": pr_auc(y_true_a, y_score_a),
        "precision": float(precision_score(y_true_a, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred, zero_division=0)),
        "predict_rate": float(y_pred.mean()),
        **counts,
    }
