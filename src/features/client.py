"""Identidad de cliente como clave de feature engineering (T-E02).

La etiqueta de este dataset esta definida a nivel de cliente, no de transaccion:
cuando una tarjeta registra fraude, se marcan como fraudulentas las transacciones
posteriores de esa tarjeta. Tratar cada transaccion como independiente ignora esa
estructura, y es la razon por la que aqui la ROC-AUC se queda en 0,9025 frente a
~0,946 de la referencia de la competencia.

Este modulo agrega las columnas C (14) por dos claves:

- **primaria** `card1 + addr1`: 95,1 % de las filas de test tienen historial.
- **refinada** `card1 + addr1 + D1n`, con `D1n = TransactionDay - D1` (el dia en
  que arranco la tarjeta): solo aporta en el 34,4 % con historial de cliente
  exacto, porque `D1n` distingue estints de uso del mismo plastico.

`D1n` es clave interna, no feature: su informacion ya esta en `D1` (la edad de
la tarjeta *es* `D1`), y emitirlo reintroduciria el identificador de cliente que
la referencia de la competencia rechaza explicitamente.

Todas las medias son **previas** al frame ordenado por `TransactionDT`, nunca
incluyen la fila actual ni filas futuras (constitution.md §3). Sin historial se
emite el centinela `SENTINEL = 0.0`, nunca `NaN`, siguiendo la convencion de
`mean_amt_card1_prev` y para no romper el check `.notna()` del pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.columns import ADDR1, CARD1, D1, DT

D1N = "D1n"

SECONDS_PER_DAY = 86400

C_COLS = [f"C{i}" for i in range(1, 15)]

#: Columnas agregadas por cliente.
#:
#: M1-M9 quedan fuera a proposito. En el dataset crudo son categoricas (`T`/`F`,
#: y M4 con tres niveles `M0`/`M1`/`M2`) y el propio preprocessor ya las declara
#: `CATEGORICAL_COLS` y las codifica con `OrdinalEncoder`, de modo que su "media
#: previa" no esta definida sobre el frame crudo que recibe `compute_features`.
#: Ademas son casi degeneradas: M1 es `T` en el 99,99 % de sus no nulos y entre el
#: 46 % y el 59 % de las M son nulas, luego su tasa previa seria casi constante y
#: calculada sobre historiales muy cortos. Las C si son recuentos numericos de
#: coincidencias, que es justo lo que promedia la referencia.
AGG_COLS = C_COLS

#: Valor emitido cuando el grupo aun no tiene historial. `NaN` esta prohibido:
#: rompe `test_inferencia_fila_unica_consistente` y el check `.notna()` del pipeline.
SENTINEL = 0.0


def transaction_day(df: pd.DataFrame) -> pd.Series:
    """Indice de dia absoluto de la transaccion."""
    return (df[DT] // SECONDS_PER_DAY).astype("float64")


def refined_uid(ordered: pd.DataFrame) -> pd.Series | None:
    """Clave de cliente refinada `card1+addr1+D1n`, o `None` si falta `D1`.

    Sin `D1` no hay forma de recuperar el dia de alta, y las filas degradan al
    nivel primario. Devolver `None` en vez de una clave de NaN hace explicita esa
    degradacion en lugar de fingir que las filas nulas comparten un mismo cliente.
    """
    if D1 not in ordered.columns:
        return None
    return (transaction_day(ordered) - ordered[D1]).rename(D1N)


def primary_keys(ordered: pd.DataFrame) -> list[pd.Series]:
    return [ordered[CARD1], ordered[ADDR1]]


def previous_count(
    ordered: pd.DataFrame, keys: list[pd.Series], name: str
) -> pd.Series:
    """Numero de filas previas del grupo, en orden temporal."""
    position = pd.Series(np.arange(len(ordered)), index=ordered.index, dtype="int64")
    return position.groupby(keys, dropna=False).cumcount().rename(name)


def previous_mean(
    ordered: pd.DataFrame, keys: list[pd.Series], col: str, name: str
) -> pd.Series:
    """Media de `col` sobre las filas **previas** del grupo.

    Definicion ingenua: `groupby(keys)[col].transform(lambda s: s.expanding().mean().shift(1))`.
    Con 35.384 grupos x 14 columnas eso son ~495.000 llamadas de Python por clave.

    Atajo vectorizado equivalente: `cumsum() - x` da la suma de filas previas sin
    necesidad de `shift`, porque la acumulacion por grupo respeta el orden
    temporal del frame. Los `NaN` de C cuentan como 0 en el numerador y no cuentan
    en el denominador, de modo que una columna ausente en una fila no destruye el
    agregado del grupo.

    `test_media_vectorizada_coincide_con_expanding` ata este atajo a la definicion
    para que la optimizacion no pueda divergir de ella.
    """
    values = ordered[col]
    filled = values.fillna(0.0).astype("float64")
    nonnull = values.notna().astype("float64")
    prior_sum = filled.groupby(keys, dropna=False).cumsum() - filled
    prior_n = nonnull.groupby(keys, dropna=False).cumsum() - nonnull
    mean = prior_sum.div(prior_n.where(prior_n > 0))
    return mean.fillna(SENTINEL).rename(name)


def _mean_family(
    ordered: pd.DataFrame, keys: list[pd.Series], suffix: str
) -> dict[str, pd.Series]:
    return {
        f"mean_{col}_prev_{suffix}": previous_mean(
            ordered, keys, col, f"mean_{col}_prev_{suffix}"
        )
        for col in AGG_COLS
        if col in ordered.columns
    }


def client_features(ordered: pd.DataFrame) -> pd.DataFrame:
    """Agregaciones historicas de cliente, sobre el frame ordenado por `TransactionDT`.

    Degrada de forma explicita: sin columnas C no hay medias, y sin `D1` no hay
    nivel refinado. Una fila con `D1` nulo conserva sus medias primarias, porque
    la clave primaria no usa `D1`.
    """
    out = pd.DataFrame(index=ordered.index)
    out = out.assign(**_mean_family(ordered, primary_keys(ordered), "client"))
    uid = refined_uid(ordered)
    if uid is not None:
        out["cnt_client_uid"] = previous_count(ordered, [uid], "cnt_client_uid")
        out = out.assign(**_mean_family(ordered, primary_keys(ordered) + [uid], "uid"))
    return out
