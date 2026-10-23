from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.categoricals import (
    NATIVE_ENCODING,
    ORDINAL_ENCODING,
    categorical_columns,
    prepare_categoricals,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card1": [-1.0, 0.0, 3.0],
            "addr1": [2.0, -1.0, 1.0],
            "ProductCD": [4.0, 0.0, 2.0],
            "V1": [1.0, 2.0, 3.0],
        }
    )


def test_columnas_categoricas_respetan_el_orden_canonico() -> None:
    assert categorical_columns(["V1", "card1", "ProductCD"]) == [
        "ProductCD",
        "card1",
    ]
    assert categorical_columns(["V1"]) == []


def test_native_desplaza_codigos_y_no_muta_el_original() -> None:
    frame = _frame()
    out = prepare_categoricals(frame, NATIVE_ENCODING)

    assert out["card1"].tolist() == [0, 1, 4]
    assert out["addr1"].tolist() == [3, 0, 2]
    assert out["ProductCD"].tolist() == [5, 1, 3]
    assert out["V1"].tolist() == [1.0, 2.0, 3.0]
    assert out["card1"].dtype == np.int32
    assert frame["card1"].tolist() == [-1.0, 0.0, 3.0]


def test_ordinal_no_modifica_la_matriz() -> None:
    frame = _frame()
    out = prepare_categoricals(frame, ORDINAL_ENCODING)
    assert out is frame
    assert out["card1"].tolist() == [-1.0, 0.0, 3.0]


def test_native_rechaza_no_finitos_y_codigos_no_enteros() -> None:
    with pytest.raises(ValueError, match="no finitos"):
        prepare_categoricals(pd.DataFrame({"card1": [0.0, np.nan]}), NATIVE_ENCODING)
    with pytest.raises(ValueError, match="no enteros"):
        prepare_categoricals(pd.DataFrame({"card1": [0.0, 1.5]}), NATIVE_ENCODING)


def test_encoding_desconocido_aborta() -> None:
    with pytest.raises(ValueError, match="Encoding categorico desconocido"):
        prepare_categoricals(_frame(), "onehot")
