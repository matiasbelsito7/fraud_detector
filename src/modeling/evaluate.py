from __future__ import annotations

import gc
import math
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

from src.modeling.dataset import SplitData
from src.modeling.folds import Fold
from src.modeling.metrics import summarize
from src.modeling.space import build_params

#: Criterio de negocio para el umbral operativo (specs.md §8.1 exige un umbral
#: "fijado segun requerimiento de negocio"). Ese requerimiento no venia
#: documentado en el repositorio, asi que se fijo explicitamente antes de tocar
#: `test`: detectar al menos el 80 % de los fraudes. Queda registrado en
#: docs/evaluation_report.md §2 junto con su justificacion y su alternativa
#: rechazada (umbral de coste relativo FP/FN).
TARGET_RECALL = 0.80

#: Grados del barrido de umbrales para el analisis de sensibilidad (specs §8.2).
SWEEP_POINTS = 201


def out_of_fold_scores(
    params: dict[str, Any],
    rounds: int,
    train: SplitData,
    folds: list[Fold],
    seed: int,
    strict: bool = False,
) -> dict[str, Any]:
    """Puntua cada fila de `train` con un modelo que no la vio en ajuste.

    Cada pliegue se ajusta en su ventana de training y puntua su ventana de
    validacion, temporal posterior. Ninguna fila se puntua con un modelo
    entrenado sobre ella, que es la condicion que hace legitimo usar estas
    puntuaciones para fijar un umbral operativo.

    `rounds` es el mismo para todos los pliegues (el del modelo final) y no el
    que resultó del early stopping en cada pliegue: así la distribución de
    puntuaciones se parece a la del modelo que se desplegará, y el umbral
    traslada. Los pliegues de la Fase H habrían usado 154, 219 y 393 rondas,
    porque ahí la ronda era parte de la medición y no del artefacto.

    Cobertura: con ventana expansiva, las primeras `n_folds * val_weeks - 1`
    semanas solo se usan para entrenar y nunca se puntúan. En `train` (19
    semanas, 3 pliegues de 3) eso deja fuera 233.881 filas: se cubren 46,1 %.
    Las filas no puntuadas se devuelven como NaN y se informa como
    `n_unscored`; con `strict=True` se exige cobertura total.
    """
    lgb_params = build_params(params, seed)
    scores = np.full(train.n_rows, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(folds):
        estimator = LGBMClassifier(n_estimators=rounds, **lgb_params)
        estimator.fit(train.X.iloc[tr], train.y.iloc[tr])
        scores[va] = np.asarray(estimator.predict_proba(train.X.iloc[va]))[:, 1]
        rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(tr)),
                "n_scored": int(len(va)),
                "rounds": int(rounds),
            }
        )
        del estimator
        gc.collect()

    n_unscored = int(np.isnan(scores).sum())
    if strict and n_unscored:
        raise ValueError(
            f"Quedaron {n_unscored} filas de train sin puntuar: los pliegues no "
            "cubren todo train y el umbral no seria representativo"
        )
    return {
        "scores": scores,
        "folds": rows,
        "n_scored": int(train.n_rows - n_unscored),
        "n_unscored": n_unscored,
        "coverage": float((train.n_rows - n_unscored) / train.n_rows),
    }


def threshold_for_target_recall(
    y_true: Any, y_score: Any, target_recall: float = TARGET_RECALL
) -> float:
    """Umbral mas bajo que alcanza `target_recall`.

    Con el recall monotonico no decreciente al bajar el umbral, el corte optimo
    es la puntuacion del k-esimo positivo mas alto, con
    `k = ceil(target_recall * n_positivos)`: incluir todos los positivos con
    puntuacion mayor o igual a ese valor garantiza el recall objetivo, y
    cualquier umbral mas bajo solo puede sumar negativos.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall debe estar en (0, 1]")
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true e y_score con distinta forma: {y.shape} vs {s.shape}")
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("Sin positivos no se puede fijar un umbral por recall")

    k = math.ceil(target_recall * n_pos)
    k = min(k, n_pos)
    pos_scores = np.sort(s[y == 1])[::-1]
    return float(pos_scores[k - 1])


def threshold_sweep(
    y_true: Any, y_score: Any, n_points: int = SWEEP_POINTS
) -> pd.DataFrame:
    """Metricas en una rejilla de umbrales, para el analisis de sensibilidad."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    grid = np.unique(np.round(np.linspace(0.0, 1.0, n_points), 6))
    rows = [
        {
            "threshold": float(t),
            **summarize(y, s, threshold=float(t)),
        }
        for t in grid
    ]
    return pd.DataFrame(rows)


def curve_points(y_true: Any, y_score: Any) -> dict[str, np.ndarray]:
    """Puntos de las curvas ROC y PR, para graficar sin recalcular metricas."""
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    fpr, tpr, roc_thresholds = roc_curve(y, s)
    precision, recall, pr_thresholds = precision_recall_curve(y, s)
    # precision_recall_curve anade un punto final (recall=0, precision=1) que no
    # tiene threshold asociado; se descarta para que las tres series midan lo mismo
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_threshold": roc_thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "pr_threshold": pr_thresholds,
    }


def recall_interval(recall: float, n_pos: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wald para un recall sobre `n_pos` positivos.

    `validation` y `test` tienen pocos miles de fraudes, asi que un recall
    puntual puede moverse varios puntos por azar de muestreo. Se reporta el
    intervalo para no leer mas precision de la que el tamano de la muestra
    soporta (constitution §6).
    """
    if n_pos <= 0:
        return (float("nan"), float("nan"))
    se = math.sqrt(max(recall * (1.0 - recall), 0.0) / n_pos)
    return (max(0.0, recall - z * se), min(1.0, recall + z * se))
