from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.modeling.dataset import SplitData
from src.modeling.folds import Fold
from src.modeling.metrics import DEFAULT_THRESHOLD, summarize
from src.modeling.space import (
    ANCHOR_PARAMS,
    SEARCH_SPACE,
    build_params,
    config_key,
    to_python,
)

#: Techo de rondas de boosting. El corte real lo hace el early stopping; este
#: techo acota el coste y mantiene el rango comparable al modelo de referencia
#: de la Fase G (600 rondas sin early stopping).
MAX_BOOST_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 30

#: `average_precision` de LightGBM == `average_precision_score` de sklearn, y es
#: exactamente la metrica de seleccion (PR-AUC), sin desajuste entre ambas.
EARLY_STOPPING_METRIC = "average_precision"
SELECTION_METRIC = "mean_pr_auc"
TIEBREAK_METRIC = "mean_roc_auc"


def build_search_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Parametros de la busqueda, con la metrica de early stopping fijada.

    Es imprescindible fijar `metric`: sin ella LightGBM evalua con
    `binary_logloss`, que no es el criterio de seleccion. Con
    `scale_pos_weight` alto el logloss se minimiza en la primera ronda y el
    early stopping corta modelos que despues siguen mejorando en PR-AUC,
    sesgando la busqueda contra la compensacion de clases.
    """
    return {**build_params(params, seed), "metric": EARLY_STOPPING_METRIC}


def tune_one(
    params: dict[str, Any], train: SplitData, folds: list[Fold], seed: int
) -> dict[str, Any]:
    """Evalua una configuracion en los pliegues walk-forward internos de `train`.

    Solo recibe `train`: ni `validation` ni `test` son accesibles desde aqui
    (constitution.md §5, tasks.md T-H01).
    """
    lgb_params = build_search_params(params, seed)
    rows = []
    started = time.perf_counter()

    for fold_id, (tr, va) in enumerate(folds):
        X_tr = train.X.iloc[tr]
        y_tr = train.y.iloc[tr]
        X_va = train.X.iloc[va]
        y_va = train.y.iloc[va]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        y_score = booster.predict(X_va, num_iteration=booster.best_iteration)
        fold_metrics = summarize(y_va, y_score, threshold=DEFAULT_THRESHOLD)
        rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "best_iteration": int(booster.best_iteration),
                "pr_auc": fold_metrics["pr_auc"],
                "roc_auc": fold_metrics["roc_auc"],
                "recall": fold_metrics["recall"],
                "precision": fold_metrics["precision"],
            }
        )
        # LightGBM 4.7 no expone free(); la memoria nativa se libera por GC.
        # Sin collect explicito los datasets de cada configuracion se acumulan y
        # el pico de memoria crece hasta agotar la RAM.
        del booster, dtrain, dvalid, X_tr, y_tr, X_va, y_va
        gc.collect()

    table = pd.DataFrame(rows)
    return {
        **{k: to_python(v) for k, v in params.items()},
        "mean_pr_auc": float(table["pr_auc"].mean()),
        "std_pr_auc": float(table["pr_auc"].std(ddof=0)),
        "min_pr_auc": float(table["pr_auc"].min()),
        "mean_roc_auc": float(table["roc_auc"].mean()),
        "mean_recall": float(table["recall"].mean()),
        "best_iteration": int(np.median(table["best_iteration"])),
        "seconds": round(time.perf_counter() - started, 1),
        "folds": rows,
    }


def run_search(
    configs: list[dict[str, Any]],
    train: SplitData,
    folds: list[Fold],
    seed: int,
    on_result: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Ejecuta la busqueda y devuelve la tabla ordenada por el criterio de seleccion."""
    anchor_key = config_key(ANCHOR_PARAMS)
    results = []
    for index, params in enumerate(configs):
        result = tune_one(params, train, folds, seed)
        result["config_id"] = index
        result["is_anchor"] = config_key(params) == anchor_key
        results.append(result)
        if on_result is not None:
            on_result(result, index + 1, len(configs))

    table = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "folds"} for r in results]
    )
    table = table.sort_values(
        [SELECTION_METRIC, TIEBREAK_METRIC], ascending=False
    ).reset_index(drop=True)
    return table, results


def select_best(table: pd.DataFrame) -> dict[str, Any]:
    """Configuracion ganadora: mayor PR-AUC medio, desempate por ROC-AUC medio."""
    if table.empty:
        raise ValueError("Tabla de busqueda vacia")
    row = table.sort_values([SELECTION_METRIC, TIEBREAK_METRIC], ascending=False).iloc[
        0
    ]
    return {
        "params": {name: to_python(row[name]) for name in SEARCH_SPACE},
        "best_iteration": int(row["best_iteration"]),
        "mean_pr_auc": float(row[SELECTION_METRIC]),
        "mean_roc_auc": float(row[TIEBREAK_METRIC]),
        "mean_recall": float(row["mean_recall"]),
        "config_id": int(row["config_id"]),
        "is_anchor": bool(row["is_anchor"]),
    }


def fit_final(
    params: dict[str, Any], best_iteration: int, train: SplitData, seed: int
) -> LGBMClassifier:
    """Reentrena la configuracion ganadora sobre todo `train`, sin validacion.

    `validation` no se usa: se entreeno con hyperparameters decididos solo con
    pliegues de `train`, asi que no habria sesgo de seleccion al incorporarla.
    Aun asi se ajusta sobre `train` unicamente para conservar una cifra limpia
    de `validation` para el informe de la Fase I.
    """
    estimator = LGBMClassifier(
        n_estimators=best_iteration, **build_params(params, seed)
    )
    estimator.fit(train.X, train.y)
    return estimator


def confirm(
    estimator: LGBMClassifier, valid: SplitData, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """Confirmacion unica de la seleccion sobre `validation`."""
    y_score = np.asarray(estimator.predict_proba(valid.X))[:, 1]
    return summarize(valid.y, y_score, threshold=threshold)
