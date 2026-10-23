from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

import numpy as np

from src.modeling.dataset import SplitData
from src.modeling.evaluate import bootstrap_ci, out_of_fold_scores
from src.modeling.folds import Fold
from src.modeling.metrics import (
    ALERT_RATES,
    pr_auc,
    roc_auc,
    summarize_at_alert_rates,
)
from src.preprocessing.categoricals import ORDINAL_ENCODING

#: Presupuesto de alerta para el que se reporta el IC bootstrap de recall.
CI_ALERT_RATE = 0.12


def _recall_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("Sin positivos no se puede calcular recall")
    return float(((y_score >= threshold) & (y_true == 1)).sum() / positives)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        aggregate[key] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
        }
    return aggregate


def evaluate_repeated(
    params: dict[str, Any],
    rounds: int,
    train: SplitData,
    folds: list[Fold],
    seeds: tuple[int, ...] = (42, 43),
    *,
    encoding: str = ORDINAL_ENCODING,
    alert_rates: tuple[float, ...] = ALERT_RATES,
    n_boot: int = 0,
) -> dict[str, Any]:
    """Evalua una configuracion en varios seeds sobre los pliegues de `train`.

    Cada seed reentrena todos los pliegues y produce scores out-of-fold. No
    recibe `validation` ni `test`: la robustez se decide dentro de `train`.
    """
    if not seeds:
        raise ValueError("seeds no puede estar vacio")
    if not folds:
        raise ValueError("folds no puede estar vacio")

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        out = out_of_fold_scores(
            params,
            rounds,
            train,
            folds,
            seed,
            encoding=encoding,
        )
        scores = np.asarray(out["scores"], dtype=float)
        mask = np.isfinite(scores)
        y = np.asarray(train.y, dtype=int)[mask]
        s = scores[mask]
        alerts = summarize_at_alert_rates(y, s, alert_rates)
        row: dict[str, Any] = {
            "seed": int(seed),
            "n_scored": int(mask.sum()),
            "coverage": float(out["coverage"]),
            "pr_auc": pr_auc(y, s),
            "roc_auc": roc_auc(y, s),
            "folds": out["folds"],
        }
        for alert in alerts:
            label = f"{float(alert['alert_rate_target']):.2f}"
            row[f"threshold_at_{label}"] = float(alert["threshold"])
            row[f"recall_at_{label}"] = float(alert["recall"])
            row[f"precision_at_{label}"] = float(alert["precision"])
            row[f"alert_rate_at_{label}"] = float(alert["predict_rate"])

        if n_boot > 0:
            row["pr_auc_ci"] = bootstrap_ci(y, s, pr_auc, n_boot=n_boot, seed=seed)
            target = min(
                alerts,
                key=lambda item: abs(float(item["alert_rate_target"]) - CI_ALERT_RATE),
            )
            threshold = float(target["threshold"])

            def recall_at_threshold(
                yy: np.ndarray, ss: np.ndarray, cut: float = threshold
            ) -> float:
                return _recall_at(yy, ss, cut)

            row["recall_ci"] = {
                "alert_rate": float(target["alert_rate_target"]),
                **bootstrap_ci(
                    y,
                    s,
                    recall_at_threshold,
                    n_boot=n_boot,
                    seed=seed,
                ),
            }
        per_seed.append(row)

    return {
        "encoding": encoding,
        "rounds": int(rounds),
        "n_folds": len(folds),
        "seeds": list(seeds),
        "alert_rates": list(alert_rates),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }
