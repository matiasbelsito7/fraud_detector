"""Importancia global e individual de las features del modelo final.

Dos lecturas complementarias, ambas sin consumir `test`:

- **gain / split**: importancia que calcula LightGBM al construir los arboles.
  No necesita datos ni valores SHAP, y es sensible a la escala de la ganancia.
- **SHAP**: contribucion de cada feature a cada prediccion concreta, aditiva y
  con signo. Se obtiene con `predict(pred_contrib=True)`, el TreeSHAP nativo de
  LightGBM, que devuelve valores SHAP exactos para modelos de arbol. No se
  anade la dependencia `shap`: no aportaria nada sobre esto (`constitution.md`
  §7) y `specs.md` §9 admite SHAP u otra tecnica equivalente.

La ultima columna que anade LightGBM a `pred_contrib` es el valor base (bias)
en el espacio de log-odds; `base + suma(contribuciones) == raw_score`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.explainability.families import feature_family

#: Nombre interno de la columna sentinela que LightGBM devuelve al final.
BASE_VALUE_COL = "__base_value__"


def _booster(estimator: Any) -> Any:
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        raise TypeError(
            "El estimador debe ser un LGBMClassifier ajustado (sin atributo booster_)"
        )
    return booster


def gain_importance(estimator: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Importancia global del booster ordenada por ganancia acumulada.

    Devuelve `rank`, `feature`, `gain`, `gain_pct`, `split` y `family`. El
    `rank` 1 es la feature que mas ganancia aporta; `split` cuenta en cuantas
    divisiones se uso, que puede ser alto aun con ganancia baja.
    """
    booster = _booster(estimator)
    names = list(feature_names)
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=int)
    if gain.size != len(names):
        raise ValueError(
            f"El booster tiene {gain.size} features y se pasaron {len(names)} nombres"
        )

    frame = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    total = float(frame["gain"].sum())
    frame["gain_pct"] = 100.0 * frame["gain"] / total if total > 0 else 0.0
    frame["family"] = frame["feature"].map(feature_family)
    frame = frame.sort_values(["gain", "feature"], ascending=[False, True])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def shap_contributions(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Matriz (n_filas, n_features + 1) de valores SHAP; la ultima es el bias."""
    values = np.asarray(estimator.predict(X, pred_contrib=True), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"pred_contrib devolvio una forma inesperada: {values.shape}")
    return values


def shap_base_value(contributions: np.ndarray) -> float:
    return float(contributions[:, -1].mean())


def shap_summary(
    estimator: Any, X: pd.DataFrame, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Importancia SHAP global: media de |SHAP| por feature, con su familia.

    `mean_abs_shap` mide cuanto mueve la feature cada prediccion; `mean_shap`
    conserva el signo medio (hacia fraude o hacia no fraude) y por eso puede
    quedar cerca de cero aunque la feature sea influyente en ambos sentidos.
    """
    contributions = shap_contributions(estimator, X)
    names = list(feature_names)
    if contributions.shape[1] != len(names) + 1:
        raise ValueError(
            f"SHAP tiene {contributions.shape[1] - 1} features y se pasaron "
            f"{len(names)} nombres"
        )
    values = contributions[:, :-1]
    frame = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    )
    frame["family"] = frame["feature"].map(feature_family)
    frame = frame.sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def aggregate_by_family(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Suma una columna de importancia por familia y calcula su porcentaje."""
    if "family" not in frame.columns:
        raise ValueError("El frame debe tener la columna 'family'")
    grouped = (
        frame.groupby("family", dropna=False)
        .agg(n_features=("feature", "size"), value=(value_col, "sum"))
        .reset_index()
        .sort_values(["value", "family"], ascending=[False, True])
    )
    total = float(grouped["value"].sum())
    grouped["pct"] = 100.0 * grouped["value"] / total if total > 0 else 0.0
    return grouped.rename(columns={"value": value_col}).reset_index(drop=True)


def local_contributions(
    estimator: Any,
    row: pd.DataFrame,
    feature_names: Sequence[str],
    top: int = 10,
) -> dict[str, Any]:
    """Explicacion de una sola transaccion: valor, bias y top contribuciones.

    `row` debe ser un DataFrame de una fila con las mismas columnas y en el
    mismo orden que el modelo. Se devuelven las `top` contribuciones por valor
    absoluto, conservando el signo: positivo empuja hacia fraude, negativo hacia
    no fraude, en log-odds.
    """
    if len(row) != 1:
        raise ValueError(f"Se esperaba una fila y llegaron {len(row)}")
    names = list(feature_names)
    if list(row.columns) != names:
        raise ValueError("Las columnas de la fila no coinciden con el modelo")
    contributions = shap_contributions(estimator, row)[0]
    values = contributions[:-1]
    order = np.argsort(np.abs(values))[::-1][:top]
    return {
        "base_value": float(contributions[-1]),
        "raw_score": float(contributions.sum()),
        "top": [
            {
                "feature": names[i],
                "value": float(row.iloc[0, i]),
                "shap": float(values[i]),
            }
            for i in order
        ],
    }


def check_additivity(estimator: Any, X: pd.DataFrame) -> float:
    """Error maximo entre `bias + suma(SHAP)` y el score crudo del modelo.

    Es la invariante que valida que los valores SHAP corresponden al estimador:
    si LightGBM cambiara la convencion, este numero dejaria de ser ~0.
    """
    contributions = shap_contributions(estimator, X)
    raw = np.asarray(estimator.predict(X, raw_score=True), dtype=float).ravel()
    return float(np.max(np.abs(contributions.sum(axis=1) - raw)))
