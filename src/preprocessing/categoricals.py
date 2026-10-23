from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.preprocessing.preprocessor import CATEGORICAL_COLS

ORDINAL_ENCODING = "ordinal"
NATIVE_ENCODING = "native"
CATEGORICAL_ENCODINGS = (ORDINAL_ENCODING, NATIVE_ENCODING)


def categorical_columns(columns: Iterable[str]) -> list[str]:
    """Columnas categoricas presentes, en el orden canonico del preprocessor."""
    present = set(columns)
    return [column for column in CATEGORICAL_COLS if column in present]


def prepare_categoricals(
    X: pd.DataFrame,
    encoding: str = ORDINAL_ENCODING,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Adapta los codigos ordinales del preprocessor al encoding del modelo.

    El preprocessor serializado asigna -1 a missing y a categorias desconocidas.
    LightGBM exige codigos enteros no negativos para tratar una columna como
    categorica nativa, asi que se desplaza cada codigo una posicion: -1 -> 0 y
    las categorias 0..N -> 1..N+1. El desplazamiento se aplica una sola vez, en
    la frontera del modelo; el parquet y el preprocessor siguen guardando el
    encoding ordinal historico.

    El encoding `ordinal` no toca la matriz y conserva la compatibilidad con los
    artefactos ya desplegados.
    """
    if encoding not in CATEGORICAL_ENCODINGS:
        raise ValueError(
            f"Encoding categorico desconocido: {encoding!r}; "
            f"validos: {list(CATEGORICAL_ENCODINGS)}"
        )
    columns = categorical_columns(X.columns)
    if encoding == ORDINAL_ENCODING or not columns:
        return X

    out = X.copy() if copy else X
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(
                f"La columna categorica {column!r} contiene valores no finitos; "
                "el preprocessor deberia haberlos codificado como -1"
            )
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"La columna categorica {column!r} contiene codigos no enteros"
            )
        out[column] = (values + 1.0).astype("int32")
    return out
