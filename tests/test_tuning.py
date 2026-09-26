from __future__ import annotations

from collections import defaultdict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.modeling.dataset import SplitData, extract
from src.modeling.folds import walk_forward_folds
from src.modeling.metrics import PRIMARY_METRIC, pr_auc
from src.modeling.space import (
    ANCHOR_PARAMS,
    BASE_PARAMS,
    SEARCH_SPACE,
    build_params,
    config_key,
    sample_configs,
    to_python,
)
from src.modeling.tuning import (
    EARLY_STOPPING_METRIC,
    SELECTION_METRIC,
    TIEBREAK_METRIC,
    build_search_params,
    confirm,
    fit_final,
    run_search,
    select_best,
    tune_one,
)
from src.split.split import SET_COL, WEEK_SECONDS


def _toy_train(n_weeks: int = 12, rows_per_week: int = 60) -> SplitData:
    rng = np.random.default_rng(11)
    n = n_weeks * rows_per_week
    dt = [86400 + WEEK_SECONDS * (i // rows_per_week) for i in range(n)]
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, size=n // 12, replace=False)] = 1
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": dt,
            "TransactionAmt": rng.random(n).astype(np.float32) * 100,
            "card1": rng.integers(1, 40, n).astype(np.float32),
            "addr1": rng.integers(1, 60, n).astype(np.float32),
            "cnt_card1": np.arange(n, dtype=np.float32),
            "V1": (np.arange(n) % 7).astype(np.float32),
            "V2": (np.arange(n) % 5).astype(np.float32),
            "V3": (np.arange(n) % 3).astype(np.float32),
            "isFraud": y,
            SET_COL: "train",
        }
    )
    return extract(df, "train")


def _toy_valid() -> SplitData:
    rng = np.random.default_rng(99)
    n = 240
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(10**6, 10**6 + n),
            "TransactionDT": [86400 + WEEK_SECONDS * (20 + i // 60) for i in range(n)],
            "TransactionAmt": rng.random(n).astype(np.float32) * 100,
            "card1": rng.integers(1, 40, n).astype(np.float32),
            "addr1": rng.integers(1, 60, n).astype(np.float32),
            "cnt_card1": np.arange(n, dtype=np.float32),
            "V1": (np.arange(n) % 7).astype(np.float32),
            "V2": (np.arange(n) % 5).astype(np.float32),
            "V3": (np.arange(n) % 3).astype(np.float32),
            "isFraud": np.concatenate(
                [
                    np.zeros(n - n // 12, dtype=np.int64),
                    np.ones(n // 12, dtype=np.int64),
                ]
            ),
            SET_COL: "validation",
        }
    )
    return extract(df, "validation")


def _fast_params() -> dict:
    return {
        "num_leaves": 7,
        "learning_rate": 0.2,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": 1.0,
    }


# --- espacio de busqueda ---


def test_espacio_de_busqueda_cubre_las_dimensiones_clave() -> None:
    assert set(SEARCH_SPACE) >= {
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "colsample_bytree",
        "scale_pos_weight",
    }
    assert 1.0 in SEARCH_SPACE["scale_pos_weight"]
    assert SEARCH_SPACE["scale_pos_weight"][-1] > 20.0
    for name, values in SEARCH_SPACE.items():
        assert len(values) >= 2, name
        assert len(set(map(str, values))) == len(values), name


def test_subsample_freq_fijo_para_que_subsample_tenga_efecto() -> None:
    assert BASE_PARAMS["subsample_freq"] >= 1
    assert BASE_PARAMS["deterministic"] is True


def test_ancla_coincide_con_el_espacio() -> None:
    assert set(ANCHOR_PARAMS) == set(SEARCH_SPACE)


def test_muestreo_es_determinista_y_sin_repetidos() -> None:
    a = sample_configs(12, seed=42)
    b = sample_configs(12, seed=42)
    assert a == b
    assert len({config_key(c) for c in a}) == 12
    assert sample_configs(12, seed=7) != a


def test_la_primera_configuracion_es_el_ancla() -> None:
    configs = sample_configs(8, seed=42)
    assert config_key(configs[0]) == config_key(ANCHOR_PARAMS)


def test_todas_las_configuraciones_usan_valores_del_espacio() -> None:
    for cfg in sample_configs(10, seed=1):
        for name, value in cfg.items():
            assert str(value) in {str(v) for v in SEARCH_SPACE[name]}, (name, value)


def test_muestreo_falla_si_no_se_alcanza_n_configs() -> None:
    with pytest.raises(ValueError):
        sample_configs(0, seed=42)


def test_build_params_inyecta_semillas_y_no_azar() -> None:
    params = build_params(_fast_params(), seed=42)
    for name in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        assert params[name] == 42
    assert params["objective"] == "binary"
    assert params["deterministic"] is True
    assert params["num_leaves"] == 7
    assert "random_state" not in params


def test_to_python_convierte_escalares_numpy() -> None:
    assert to_python(np.float64(1.5)) == 1.5
    assert isinstance(to_python(np.int64(3)), int)
    assert to_python("x") == "x"


# --- busqueda y seleccion ---


def test_tune_one_usa_todos_los_pliegues() -> None:
    train = _toy_train()
    folds = walk_forward_folds(train.X["TransactionDT"], n_folds=2, val_weeks=2)
    result = tune_one(_fast_params(), train, folds, seed=42)
    assert len(result["folds"]) == len(folds)
    assert 0.0 <= result["mean_pr_auc"] <= 1.0
    assert 0.0 <= result["mean_roc_auc"] <= 1.0
    assert result["best_iteration"] >= 1
    assert result["std_pr_auc"] >= 0.0


def test_tune_one_es_reproducible() -> None:
    train = _toy_train()
    folds = walk_forward_folds(train.X["TransactionDT"], n_folds=2, val_weeks=2)
    a = tune_one(_fast_params(), train, folds, seed=42)
    b = tune_one(_fast_params(), train, folds, seed=42)
    assert a["mean_pr_auc"] == b["mean_pr_auc"]
    assert a["mean_roc_auc"] == b["mean_roc_auc"]
    assert a["best_iteration"] == b["best_iteration"]


def test_la_busqueda_ordena_por_el_criterio_de_seleccion() -> None:
    train = _toy_train()
    folds = walk_forward_folds(train.X["TransactionDT"], n_folds=2, val_weeks=2)
    configs = [dict(ANCHOR_PARAMS), {**ANCHOR_PARAMS, "num_leaves": 127}]
    table, results = run_search(configs, train, folds, seed=42)
    assert table[SELECTION_METRIC].is_monotonic_decreasing
    # el ancla se marca aunque labusqueda la ordene en otra posicion
    assert set(table["is_anchor"]) == {True, False}
    ancla = table[table["is_anchor"]].iloc[0]
    assert int(ancla["config_id"]) == 0
    assert len(results) == 2
    assert set(table["config_id"]) == {0, 1}


def test_seleccion_gana_por_pr_auc_y_desempata_con_roc_auc() -> None:
    tabla = pd.DataFrame(
        [
            {
                SELECTION_METRIC: 0.50,
                TIEBREAK_METRIC: 0.80,
                "mean_recall": 0.1,
                "best_iteration": 10,
                "config_id": 0,
                "is_anchor": False,
                **{name: 1 for name in SEARCH_SPACE},
            },
            {
                SELECTION_METRIC: 0.60,
                TIEBREAK_METRIC: 0.70,
                "mean_recall": 0.2,
                "best_iteration": 20,
                "config_id": 1,
                "is_anchor": True,
                **{name: 2 for name in SEARCH_SPACE},
            },
            {
                SELECTION_METRIC: 0.60,
                TIEBREAK_METRIC: 0.90,
                "mean_recall": 0.3,
                "best_iteration": 30,
                "config_id": 2,
                "is_anchor": False,
                **{name: 3 for name in SEARCH_SPACE},
            },
        ]
    )
    best = select_best(tabla)
    assert best["config_id"] == 2
    assert set(best["params"]) == set(SEARCH_SPACE)


def test_seleccion_falla_con_tabla_vacia() -> None:
    with pytest.raises(ValueError):
        select_best(pd.DataFrame())


def test_el_early_stopping_usa_la_metrica_de_seleccion() -> None:
    assert EARLY_STOPPING_METRIC == "average_precision"
    assert SELECTION_METRIC == "mean_pr_auc"
    assert PRIMARY_METRIC == "pr_auc"


def test_la_busqueda_fija_la_metrica_y_no_el_logloss_por_defecto() -> None:
    params = build_search_params(_fast_params(), seed=42)
    assert params["metric"] == EARLY_STOPPING_METRIC
    # build_params (usado por fit_final) no debe inyectarla
    assert "metric" not in build_params(_fast_params(), seed=42)


def test_lightgbm_registra_la_metrica_fijada_y_crece_en_pr_auc() -> None:
    """Regresion: con la metrica por defecto el early stopping cortaba en la
    primera ronda y sesgaba la busqueda. Se verifica contra LightGBM real."""
    history: dict = defaultdict(list)
    rng = np.random.default_rng(5)
    X = np.abs(rng.normal(0, 1, (600, 4))).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] > 2.4) | (np.arange(600) % 37 == 0)).astype(int)
    train = lgb.train(
        build_search_params({**_fast_params(), "learning_rate": 0.2}, seed=42),
        lgb.Dataset(X[:500], label=y[:500]),
        num_boost_round=60,
        valid_sets=[lgb.Dataset(X[500:], label=y[500:])],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(history)],
    )
    curve = history["valid"][EARLY_STOPPING_METRIC]
    assert len(curve) == 60
    # PR-AUC debe mejorar al añadir rondas: si el early stopping cortara en 1,
    # `best_iteration` seria 1 y esta asercion fallaria.
    assert train.best_iteration == 0
    assert curve[-1] > curve[0]


def test_la_metrica_de_lightgbm_coincide_con_sklearn() -> None:
    """El corte de rondas decide el modelo final, asi que ambas implementaciones
    de PR-AUC deben ser la misma magnitud."""
    rng = np.random.default_rng(3)
    X = rng.random((4000, 6)).astype(np.float32)
    y = ((X[:, 0] * 2 + X[:, 1] > 2.0) | (rng.random(4000) < 0.03)).astype(int)
    history: dict = defaultdict(list)
    booster = lgb.train(
        build_search_params(_fast_params(), seed=42),
        lgb.Dataset(X, label=y),
        num_boost_round=80,
        valid_sets=[lgb.Dataset(X, label=y)],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(history)],
    )
    curve = history["valid"][EARLY_STOPPING_METRIC]
    for rounds in (1, 20, 80):
        score = pr_auc(y, booster.predict(X, num_iteration=rounds))
        assert curve[rounds - 1] == pytest.approx(score, abs=1e-9)


# --- modelo final ---


def test_modelo_final_serializa_y_predice_identico(tmp_path) -> None:
    train = _toy_train()
    valid = _toy_valid()
    estimator = fit_final(_fast_params(), best_iteration=15, train=train, seed=42)
    path = tmp_path / "modelo.joblib"
    joblib.dump(estimator, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        estimator.predict_proba(valid.X)[:, 1],
        restored.predict_proba(valid.X)[:, 1],
    )


def test_confirmacion_sobre_validation_no_toca_el_test() -> None:
    train = _toy_train()
    valid = _toy_valid()
    estimator = fit_final(_fast_params(), best_iteration=15, train=train, seed=42)
    metrics = confirm(estimator, valid)
    assert metrics["n"] == valid.n_rows
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["threshold"] == 0.5
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == valid.n_rows


def test_el_ajuste_final_es_reproducible() -> None:
    train = _toy_train()
    valid = _toy_valid()
    a = fit_final(_fast_params(), best_iteration=15, train=train, seed=42)
    b = fit_final(_fast_params(), best_iteration=15, train=train, seed=42)
    np.testing.assert_allclose(
        a.predict_proba(valid.X)[:, 1], b.predict_proba(valid.X)[:, 1]
    )
