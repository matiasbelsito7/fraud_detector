from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src.modeling.dataset import SplitData
from src.modeling.folds import walk_forward_folds
from src.modeling.robustness import evaluate_repeated
from src.modeling.space import BASE_PARAMS
from src.preprocessing.categoricals import NATIVE_ENCODING, ORDINAL_ENCODING


def _toy() -> SplitData:
    rng = np.random.default_rng(21)
    weeks = 12
    rows_per_week = 60
    n = weeks * rows_per_week
    dt = np.repeat(np.arange(weeks) * 604_800, rows_per_week)
    f0 = rng.normal(size=n).astype("float32")
    X = pd.DataFrame(
        {
            "TransactionDT": dt,
            "f0": f0,
            "f1": rng.normal(size=n).astype("float32"),
            "card1": (np.arange(n) % 23).astype("float32"),
            "addr1": (np.arange(n) % 13).astype("float32"),
        }
    )
    y = pd.Series((f0 + rng.normal(scale=0.5, size=n) > 0.7).astype("int8"))
    return SplitData(name="train", X=X, y=y)


def _params() -> dict[str, float]:
    return {**BASE_PARAMS, "learning_rate": 0.3, "scale_pos_weight": 1.0}


def test_evaluacion_repetida_ordinal_y_nativa() -> None:
    train = _toy()
    folds = walk_forward_folds(train.X["TransactionDT"], n_folds=2, val_weeks=2)
    for encoding in (ORDINAL_ENCODING, NATIVE_ENCODING):
        result = evaluate_repeated(
            _params(),
            25,
            train,
            folds,
            seeds=(42, 43),
            encoding=encoding,
            n_boot=0,
        )
        assert result["encoding"] == encoding
        assert result["n_folds"] == 2
        assert len(result["per_seed"]) == 2
        assert result["per_seed"][0]["seed"] == 42
        assert len(result["per_seed"][0]["folds"]) == 2
        assert result["aggregate"]["pr_auc"]["mean"] > 0.0
        assert result["aggregate"]["recall_at_0.12"]["mean"] >= 0.0


def test_evaluacion_repetida_no_recibe_validation_ni_test() -> None:
    parameters = inspect.signature(evaluate_repeated).parameters
    assert "validation" not in parameters
    assert "test" not in parameters
