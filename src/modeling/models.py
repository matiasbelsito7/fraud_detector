from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42

#: Fila maxima de entrenamiento para los modelos lineales y de arboles.
#: 762 features en float64 sobre 434k filas ocupan ~2.6 GB; el limite acota el pico
#: de memoria manteniendo una submuestra representativa (la tasa de fraude es 3.5%).
MAX_ROWS_CLASSICAL = 120_000

FAMILY_BASELINE = "baseline"
FAMILY_CLASSICAL = "classical"
FAMILY_BOOSTING = "boosting"

FAMILY_ORDER = (FAMILY_BASELINE, FAMILY_CLASSICAL, FAMILY_BOOSTING)


@dataclass(frozen=True)
class Candidate:
    """Modelo candidato con su familia y parametros fijos (sin tuning: Fase H)."""

    name: str
    family: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)
    max_rows: int | None = None
    note: str = ""


def _lgbm(seed: int, **overrides: Any) -> LGBMClassifier:
    params: dict[str, Any] = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_lambda": 1.0,
        "objective": "binary",
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        # Reproducibilidad: histogramas deterministas, sin dependencia del hardware
        "deterministic": True,
        "force_row_wise": True,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def build_candidates(
    seed: int = SEED, pos_weight: float | None = None
) -> list[Candidate]:
    """Escalera de modelos de specs.md §7: baseline -> clasicos -> boosting.

    Los parametros son referencias fijas, no resultados de tuning (Fase H).
    `pos_weight` activa la variante que compensa el desbalance de clases; si es
    None no se genera esa variante.
    """
    candidates = [
        Candidate(
            name="dummy_prior",
            family=FAMILY_BASELINE,
            estimator=DummyClassifier(strategy="prior"),
            params={"strategy": "prior"},
            note="Suelo honesto: predice la tasa de fraude de train sin usar features.",
        ),
        Candidate(
            name="logreg",
            family=FAMILY_CLASSICAL,
            estimator=LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
            ),
            params={"solver": "lbfgs", "C": 1.0, "max_iter": 1000},
            max_rows=MAX_ROWS_CLASSICAL,
            note="Referencia lineal sobre submuestra; baseline con senal.",
        ),
        Candidate(
            name="random_forest",
            family=FAMILY_CLASSICAL,
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            params={
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 50,
                "class_weight": "balanced_subsample",
            },
            max_rows=MAX_ROWS_CLASSICAL,
            note="Referencia de arboles con bagging y desbalanceo por clase.",
        ),
        Candidate(
            name="lgbm",
            family=FAMILY_BOOSTING,
            estimator=_lgbm(seed),
            params={
                "n_estimators": 600,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "colsample_bytree": 0.6,
            },
            note="Boosting de gradiente, referencia sin reponderacion de clases.",
        ),
    ]
    if pos_weight is not None:
        candidates.append(
            Candidate(
                name="lgbm_pos_weight",
                family=FAMILY_BOOSTING,
                estimator=_lgbm(seed, scale_pos_weight=pos_weight),
                params={
                    "n_estimators": 600,
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "colsample_bytree": 0.6,
                    "scale_pos_weight": pos_weight,
                },
                note=(
                    "Igual que lgbm pero compensando el desbalance "
                    f"neg/pos={pos_weight:.1f} calculado sobre train."
                ),
            )
        )
    return sorted(candidates, key=lambda c: FAMILY_ORDER.index(c.family))
