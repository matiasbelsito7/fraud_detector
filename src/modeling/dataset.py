from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocessing.preprocessor import ID_COLS, TARGET_COL
from src.split.split import SET_COL, SETS

NON_FEATURE_COLS = [*ID_COLS, TARGET_COL, SET_COL]

TRAIN_SET = "train"
VAL_SET = "validation"
TEST_SET = "test"

EVAL_SETS = (TRAIN_SET, VAL_SET)


@dataclass(frozen=True)
class SplitData:
    """X/y de un conjunto del split temporal."""

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def n_rows(self) -> int:
        return int(len(self.X))

    @property
    def pos_rate(self) -> float:
        return float(self.y.mean())


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas usables como features: se excluyen id, target y etiqueta de split.

    `TransactionDT` se conserva: es passthrough del preprocessing y esta
    disponible en inferencia (una transaccion nueva trae su propio timestamp).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def load_split(path: str | Path) -> pd.DataFrame:
    """Carga `split.parquet` y valida que tenga las columnas de control."""
    df = pd.read_parquet(path)
    missing = [c for c in NON_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas ausentes: {missing}")
    return df


def extract(df: pd.DataFrame, name: str) -> SplitData:
    if name not in SETS:
        raise ValueError(f"Conjunto desconocido: {name}")
    mask = df[SET_COL] == name
    if not mask.any():
        raise ValueError(f"Conjunto vacio: {name}")
    index = df.index[mask]
    features = feature_columns(df)
    return SplitData(
        name=name,
        X=df.loc[index, features],
        y=df.loc[index, TARGET_COL].astype("int8"),
    )


def load_splits(
    path: str | Path, names: tuple[str, ...] = EVAL_SETS
) -> dict[str, SplitData]:
    """Carga solo los conjuntos indicados. Por defecto excluye `test` (aislamiento)."""
    df = load_split(path)
    return {name: extract(df, name) for name in names}
