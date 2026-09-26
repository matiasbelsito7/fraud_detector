from __future__ import annotations

from typing import Any

import numpy as np

#: Parametros fijos: identifican el problema, no se buscan.
BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    # sin `subsample_freq` > 0 el submuestreo de filas no tiene efecto
    "subsample_freq": 1,
    "verbose": -1,
    # histogramas deterministas: resultado independiente del numero de hilos
    "deterministic": True,
    "force_row_wise": True,
}

#: Semillas que LightGBM exige para que todo el azar sea reproducible.
SEED_PARAMS = ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed")

#: Espacio de busqueda (specs.md §7.4). Listas discretas y no rangos continuos:
#: con 434k filas y 762 features cada evaluacion cuesta minutos, asi que el
#: presupuesto de 20 configuraciones no cubre un grid denso.
SEARCH_SPACE: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63, 127, 255],
    # Sin 0.02: con 434k filas necesitaria >2000 rondas por configuracion, muy
    # por encima del presupuesto de computo de esta fase.
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_samples": [20, 100, 500, 2000],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "subsample": [0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0, 100.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    # 1.0 = sin compensacion; 27.5 = compensacion del desbalance medido en train.
    # El tuning decide si compensar mejora, en vez de asumirlo.
    "scale_pos_weight": [1.0, 27.5],
}

#: Ancla: configuracion exacta del modelo `lgbm` de la Fase G
#: (`src/modeling/models.py::_lgbm`). Si el tuning no la supera, la Fase G ya
#: era optima y eso se reporta como resultado negativo (constitution.md §6).
ANCHOR_PARAMS: dict[str, Any] = {
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 100,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "scale_pos_weight": 1.0,
}


def to_python(value: Any) -> Any:
    """Convierte escalares de NumPy a tipos de Python (JSON y LightGBM)."""
    return value.item() if isinstance(value, np.generic) else value


def build_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Combina parametros buscados, fijos y semillas en un dict de LightGBM."""
    out: dict[str, Any] = {
        **BASE_PARAMS,
        **{k: to_python(v) for k, v in params.items()},
    }
    for name in SEED_PARAMS:
        out[name] = seed
    out["num_threads"] = -1
    return out


def config_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Clave canonica de una configuracion, para deduplicar y comparar."""
    return tuple(sorted((k, str(to_python(v))) for k, v in params.items()))


def sample_configs(
    n_configs: int, seed: int, anchor: dict[str, Any] | None = ANCHOR_PARAMS
) -> list[dict[str, Any]]:
    """Random search determinista; la configuracion 0 es siempre el ancla.

    Misma semilla -> misma lista de configuraciones, incluidas las repeticiones
    descartadas, de modo que la busqueda completa es reproducible.
    """
    if n_configs < 1:
        raise ValueError("n_configs debe ser >= 1")

    rng = np.random.default_rng(seed)
    seen: set[tuple[tuple[str, str], ...]] = set()
    configs: list[dict[str, Any]] = []

    if anchor is not None:
        configs.append({k: to_python(v) for k, v in anchor.items()})
        seen.add(config_key(anchor))

    attempts = 0
    max_attempts = n_configs * 200
    while len(configs) < n_configs and attempts < max_attempts:
        attempts += 1
        candidate = {
            name: to_python(rng.choice(values)) for name, values in SEARCH_SPACE.items()
        }
        key = config_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)

    if len(configs) < n_configs:
        raise ValueError(
            f"No se pudieron muestrear {n_configs} configuraciones distintas; "
            f"espacio insuficiente tras {attempts} intentos"
        )
    return configs
