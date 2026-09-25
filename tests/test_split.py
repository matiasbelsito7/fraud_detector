from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.split.split import (
    SETS,
    TRAIN_WEEKS,
    VAL_WEEKS,
    WEEK_SECONDS,
    apply_split,
    assign_sets,
    check_summary,
    summary,
    week_index,
)


def _toy_df(n_weeks: int = 30) -> pd.DataFrame:
    n = n_weeks * 10
    dt = [86400 + WEEK_SECONDS * (i // 10) for i in range(n)]
    return pd.DataFrame(
        {
            "TransactionDT": dt,
            "isFraud": [1 if i % 9 == 0 else 0 for i in range(n)],
        }
    )


def test_week_index_determinista_y_esperado() -> None:
    df = _toy_df()
    wk = week_index(df["TransactionDT"])
    assert wk.iloc[0] == 0
    assert wk.iloc[-1] == 29
    assert int(wk.max()) + 1 == 30


def test_division_completa_y_disjunta() -> None:
    df = _toy_df()
    labels = assign_sets(df["TransactionDT"])
    assert labels.isin(SETS).all()
    assert labels.nunique() == len(SETS)
    assert labels.index.equals(df.index)


def test_orden_temporal_estricto() -> None:
    df = _toy_df()
    out = apply_split(df)
    checks = check_summary(out)
    ordering = [c for c in checks if "temporal" in c["check"]][0]
    assert ordering["pass"]


def test_semanas_por_conjunto() -> None:
    labels = assign_sets(_toy_df()["TransactionDT"])
    assert (labels == "train").sum() == TRAIN_WEEKS * 10
    assert (labels == "validation").sum() == VAL_WEEKS * 10
    assert (labels == "test").sum() == (30 - TRAIN_WEEKS - VAL_WEEKS) * 10


def test_validacion_reproducible() -> None:
    df = _toy_df()
    checks = check_summary(apply_split(df))
    assert all(c["pass"] for c in checks)


def test_apply_split_determinista() -> None:
    df = _toy_df()
    a = apply_split(df)
    b = apply_split(df)
    pdt.assert_frame_equal(a, b)


def _labels_by_dt(df: pd.DataFrame) -> pd.Series:
    return (
        df.assign(set_=assign_sets(df["TransactionDT"]))
        .sort_values("TransactionDT")["set_"]
        .reset_index(drop=True)
    )


def test_array_split_inquietudes_sin_memoria_del_orden() -> None:
    df = _toy_df()
    shuffled = df.sample(frac=1, random_state=7)
    pdt.assert_series_equal(_labels_by_dt(df), _labels_by_dt(shuffled))


def test_checks_detectan_desorden() -> None:
    df = _toy_df()
    out = apply_split(df)
    out.loc[out["TransactionDT"].idxmin(), "split_set"] = "test"
    checks = check_summary(out)
    # El orden temporal deja de ser estricto o la division deja de ser valida
    assert not all(c["pass"] for c in checks)


def test_resumen_contiene_tres_sets() -> None:
    out = apply_split(_toy_df())
    summ = summary(out)
    assert set(summ["set"]) == set(SETS)
    assert np.isclose(summ["pct"].sum(), 100.0)
