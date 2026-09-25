from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.preprocessing.preprocessor import (
    get_roles,
    preprocess_dataset,
    transform,
)


def _toy_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "isFraud": [0, 1, 0, 0, 1, 0],
            "TransactionDT": [86400 + i * 100 for i in range(n)],
            "TransactionAmt": [10.0, 20.0, 5.0, 100.0, 50.0, 200.0],
            "V1": [0.1, np.nan, 0.5, -0.2, np.nan, 0.9],
            "D7": [10.0] * n,
            "dist2": [1.0] * n,
            "ProductCD": ["W", "C", np.nan, "W", "H", np.nan],
            "card1": [7919, 9500, 7919, 15885, 9500, 7919],
        }
    )


def test_roles_excluyen_drop_y_asignan_roles() -> None:
    df = _toy_df()
    roles = get_roles(df)
    assert roles.numeric == ["V1"]
    assert roles.amt == ["TransactionAmt"]
    assert roles.categorical == ["ProductCD", "card1"]
    assert roles.passthrough == ["TransactionDT"]
    assert roles.drop == ["dist2", "D7"]


def test_roles_faltante_requerida_levanta_error() -> None:
    df = _toy_df().drop(columns=["TransactionAmt"])
    try:
        get_roles(df)
    except ValueError as exc:
        assert "TransactionAmt" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError por columna requerida ausente")


def test_preprocesado_determinista() -> None:
    df = _toy_df()
    _roles, pipeline, out_1 = preprocess_dataset(df)
    _roles_2, _pipe_2, out_2 = preprocess_dataset(df, pipeline=pipeline)
    pdt.assert_frame_equal(out_1, out_2)


def test_columnas_dropeadas_ausentes_del_resultado() -> None:
    _roles, _pipeline, out = preprocess_dataset(_toy_df())
    assert "D7" not in out.columns
    assert "dist2" not in out.columns
    assert "TransactionID" in out.columns
    assert "isFraud" in out.columns


def test_nan_conserva_indicador_y_sin_valores_faltantes() -> None:
    df = _toy_df()
    roles, pipeline, _out = preprocess_dataset(df)
    features = transform(df, pipeline, roles)
    assert features.isna().sum().sum() == 0
    assert len(features.columns) == 6


def test_nan_en_categorica_se_encoda() -> None:
    df = _toy_df()
    _roles, pipeline, out = preprocess_dataset(df)
    assert out.loc[2, "ProductCD"] == -1
    assert out.loc[5, "ProductCD"] == -1


def test_categoria_desconocida_en_inferencia() -> None:
    df = _toy_df()
    infer = df.copy()
    infer["ProductCD"] = ["Z"] * len(infer)
    roles, pipeline, _out = preprocess_dataset(df)
    features = transform(infer, pipeline, roles)
    assert (features["ProductCD"] == -1).all()
