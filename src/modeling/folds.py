from __future__ import annotations

import numpy as np
import pandas as pd

from src.split.split import week_index

N_FOLDS = 3
VAL_WEEKS = 3

Fold = tuple[np.ndarray, np.ndarray]


def walk_forward_folds(
    dt: pd.Series, n_folds: int = N_FOLDS, val_weeks: int = VAL_WEEKS
) -> list[Fold]:
    """Pliegues temporales de ventana expansiva dentro de `train`.

    Devuelve indices posicionales de la media ventana y de la ventana de
    validacion, en orden temporal. El training siempre precede a la validacion,
    de modo que ningun pliegue ve informacion futura (constitution §3).

    Con `train` = 18 semanas y 3 pliegues de 3 semanas:
    (sem 0-8 / 9-11), (sem 0-11 / 12-14), (sem 0-14 / 15-17).
    """
    if n_folds < 1 or val_weeks < 1:
        raise ValueError("n_folds y val_weeks deben ser >= 1")

    wk = week_index(dt).to_numpy()
    total = int(wk.max()) + 1
    first_val = total - n_folds * val_weeks
    if first_val < 1:
        raise ValueError(
            f"Datos insuficientes: {total} semanas no alcanzan para "
            f"{n_folds} pliegues de {val_weeks} semanas con training inicial"
        )

    folds: list[Fold] = []
    for k in range(n_folds):
        v0 = first_val + k * val_weeks
        v1 = v0 + val_weeks
        tr = np.flatnonzero(wk < v0)
        va = np.flatnonzero((wk >= v0) & (wk < v1))
        folds.append((tr, va))
    return folds


def fold_summary(dt: pd.Series, y: pd.Series, folds: list[Fold]) -> pd.DataFrame:
    """Descripcion por pliegue: filas, semanas y tasa de fraude de cada ventana."""
    wk = week_index(dt)
    rows = []
    for i, (tr, va) in enumerate(folds):
        rows.append(
            {
                "fold": i,
                "n_train": int(len(tr)),
                "n_valid": int(len(va)),
                "sem_train_max": int(wk.iloc[tr].max()),
                "sem_valid_min": int(wk.iloc[va].min()),
                "sem_valid_max": int(wk.iloc[va].max()),
                "fraude_pct_valid": round(
                    100.0 * float(np.asarray(y).astype(int)[va].mean()), 3
                ),
            }
        )
    return pd.DataFrame(rows)


def check_folds(dt: pd.Series, folds: list[Fold]) -> list[dict]:
    """Valida que los pliegues sean temporales, no vacios y disjuntos en validacion."""
    wk = week_index(dt)
    checks: list[dict] = []

    non_empty = all(len(tr) > 0 and len(va) > 0 for tr, va in folds)
    checks.append(
        {
            "check": "Pliegues no vacios",
            "pass": non_empty,
            "detail": f"n_folds={len(folds)}",
        }
    )

    if non_empty:
        temporal = all(
            int(wk.iloc[tr].max()) < int(wk.iloc[va].min()) for tr, va in folds
        )
        detail = "; ".join(
            f"f{i}: sem<={int(wk.iloc[tr].max())} < sem>={int(wk.iloc[va].min())}"
            for i, (tr, va) in enumerate(folds)
        )
    else:
        temporal, detail = False, "pliegues vacios: no evaluable"
    checks.append(
        {
            "check": "Training precede a validacion en cada pliegue",
            "pass": bool(temporal),
            "detail": detail,
        }
    )

    valid_blocks = [set(va.tolist()) for _, va in folds]
    disjoint = all(
        valid_blocks[i].isdisjoint(valid_blocks[j])
        for i in range(len(valid_blocks))
        for j in range(i + 1, len(valid_blocks))
    )
    checks.append(
        {
            "check": "Ventanas de validacion disjuntas",
            "pass": bool(disjoint),
            "detail": f"ventanas={len(valid_blocks)}",
        }
    )

    growing = all(
        len(folds[i][0]) < len(folds[i + 1][0]) for i in range(len(folds) - 1)
    )
    checks.append(
        {
            "check": "Ventana de training expansiva",
            "pass": bool(growing and non_empty),
            "detail": " > ".join(str(len(tr)) for tr, _ in folds),
        }
    )
    return checks
