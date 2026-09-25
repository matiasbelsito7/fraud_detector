from __future__ import annotations

import numpy as np
import pandas as pd

DT_COL = "TransactionDT"
SET_COL = "split_set"

WEEK_SECONDS = 7 * 24 * 3600
TRAIN_WEEKS = 18
VAL_WEEKS = 4
TARGET = "isFraud"

SETS = ("train", "validation", "test")


def week_index(dt: pd.Series) -> pd.Series:
    """Indice de semana (0..N) desde el minimo de TransactionDT."""
    return ((dt - dt.min()) // WEEK_SECONDS).astype(np.int64)


def assign_sets(dt: pd.Series) -> pd.Series:
    """Asigna train/validation/test por semana temporal (determinista, sin semilla)."""
    wk = week_index(dt)
    labels = np.where(
        wk < TRAIN_WEEKS,
        "train",
        np.where(wk < TRAIN_WEEKS + VAL_WEEKS, "validation", "test"),
    )
    return pd.Series(labels.astype(str), index=dt.index, name=SET_COL)


def summary(df: pd.DataFrame, set_col: str = SET_COL) -> pd.DataFrame:
    wk_all = week_index(df[DT_COL])
    rows = []
    for label, sub in df.groupby(set_col):
        pos = int(sub[TARGET].sum()) if TARGET in sub.columns else -1
        wk_sub = wk_all[sub.index]
        rows.append(
            {
                "set": label,
                "n": int(len(sub)),
                "pct": round(100.0 * len(sub) / max(len(df), 1), 2),
                "fraude_n": pos,
                "fraude_pct": round(100.0 * pos / max(len(sub), 1), 3),
                "dt_min": int(sub[DT_COL].min()),
                "dt_max": int(sub[DT_COL].max()),
                "semana_min": int(wk_sub.min()) if len(sub) else None,
                "semana_max": int(wk_sub.max()) if len(sub) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("semana_min")


def check_summary(df: pd.DataFrame, set_col: str = SET_COL) -> list[dict]:
    checks = []
    dt = df[DT_COL]
    sets = df[set_col]

    checks.append(
        {
            "check": "Division completa (todas las filas asignadas)",
            "pass": bool(sets.isin(SETS).all()),
            "detail": f"valores={sorted(sets.unique().tolist())}",
        }
    )
    sub = {s: df[df[set_col] == s] for s in SETS}
    empty = [s for s in SETS if sub[s].empty]
    checks.append(
        {
            "check": "Ninguno de los tres conjuntos vacio",
            "pass": not empty,
            "detail": f"vacios={empty}",
        }
    )
    if TARGET in df.columns:
        missing_class = [
            s
            for s in SETS
            if not sub[s].empty and sorted(sub[s][TARGET].unique().tolist()) != [0, 1]
        ]
        checks.append(
            {
                "check": "Ambas clases presentes en cada conjunto",
                "pass": not missing_class,
                "detail": f"conjuntos sin ambas clases={missing_class}",
            }
        )

    wk = week_index(dt)
    wk_by_set = {s: wk[df[set_col] == s] for s in SETS}
    ordering = (
        wk_by_set["train"].max()
        < wk_by_set["validation"].min()
        <= wk_by_set["validation"].max()
        < wk_by_set["test"].min()
    )
    checks.append(
        {
            "check": "Orden temporal estricto (train < validation < test)",
            "pass": bool(ordering),
            "detail": (
                f"train sem={int(wk_by_set['train'].min())}-{int(wk_by_set['train'].max())} "
                f"val sem={int(wk_by_set['validation'].min())}-{int(wk_by_set['validation'].max())} "
                f"test sem={int(wk_by_set['test'].min())}-{int(wk_by_set['test'].max())}"
            ),
        }
    )
    return checks


def apply_split(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve df + columna split_set (disjunta, completa y ordenada)."""
    out = df.copy()
    if SET_COL in out.columns:
        out = out.drop(columns=[SET_COL])
    out[SET_COL] = assign_sets(out[DT_COL])
    return out
