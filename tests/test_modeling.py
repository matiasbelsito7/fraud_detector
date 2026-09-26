from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.modeling.dataset import (
    EVAL_SETS,
    NON_FEATURE_COLS,
    TEST_SET,
    TRAIN_SET,
    VAL_SET,
    extract,
    feature_columns,
)
from src.modeling.folds import (
    check_folds,
    fold_summary,
    walk_forward_folds,
)
from src.modeling.metrics import confusion_counts, pr_auc, roc_auc, summarize
from src.modeling.models import (
    FAMILY_BASELINE,
    FAMILY_BOOSTING,
    FAMILY_CLASSICAL,
    Candidate,
    build_candidates,
)
from src.modeling.train import neg_pos_ratio, run_candidates, subsample
from src.split.split import SET_COL, WEEK_SECONDS, week_index


def _toy_split(n_weeks: int = 30, rows_per_week: int = 12) -> pd.DataFrame:
    n = n_weeks * rows_per_week
    rng = np.random.default_rng(0)
    dt = [86400 + WEEK_SECONDS * (i // rows_per_week) for i in range(n)]
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n).astype(np.int64),
            "TransactionDT": dt,
            "TransactionAmt": rng.random(n).astype(np.float32) * 100,
            "card1": rng.integers(1, 5, n).astype(np.float32),
            "cnt_card1": np.arange(n, dtype=np.float32),
            SET_COL: ["train"] * 60 + ["validation"] * 20 + ["test"] * (n - 80),
        }
    )


def _toy_split_data(n_weeks: int = 18, rows_per_week: int = 40):
    rng = np.random.default_rng(7)
    n_tr = n_weeks * rows_per_week
    n_va = 4 * rows_per_week
    dt_tr = [86400 + WEEK_SECONDS * (i // rows_per_week) for i in range(n_tr)]
    dt_va = [86400 + WEEK_SECONDS * (n_weeks + i // rows_per_week) for i in range(n_va)]
    feats = {
        "TransactionDT": dt_tr + dt_va,
        "TransactionAmt": rng.random(n_tr + n_va).astype(np.float32) * 100,
        "card1": rng.integers(1, 30, n_tr + n_va).astype(np.float32),
        "cnt_card1": np.arange(n_tr + n_va, dtype=np.float32),
    }
    y = np.zeros(n_tr + n_va, dtype=np.int64)
    y[::11] = 1
    df = pd.DataFrame(
        {**feats, "isFraud": y, SET_COL: ["train"] * n_tr + ["validation"] * n_va}
    )
    return extract(df, TRAIN_SET), extract(df, VAL_SET)


def _toy_candidate(name: str = "dummy", max_rows: int | None = None) -> Candidate:
    return Candidate(
        name=name,
        family=FAMILY_BASELINE,
        estimator=DummyClassifier(strategy="prior"),
        params={"strategy": "prior"},
        max_rows=max_rows,
    )


# --- dataset ---


def test_columnas_excluyen_id_target_y_split() -> None:
    df = _toy_split()
    feats = feature_columns(df)
    assert not set(feats) & set(NON_FEATURE_COLS)
    assert set(NON_FEATURE_COLS) == {"TransactionID", "isFraud", SET_COL}
    assert "TransactionDT" in feats


def test_extract_separa_y_conserva_filas() -> None:
    df = _toy_split()
    train = extract(df, TRAIN_SET)
    val = extract(df, VAL_SET)
    assert train.n_rows == 60
    assert val.n_rows == 20
    assert train.X.index.intersection(val.X.index).empty
    assert set(train.X.columns) == set(feature_columns(df))
    assert train.y.dtype == np.int8
    assert 0.0 < train.pos_rate < 1.0


def test_extract_falla_con_conjunto_desconocido_o_vacio() -> None:
    df = _toy_split()
    try:
        extract(df, "otro")
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError por conjunto desconocido")
    sin_val = df.drop(index=df.index[df[SET_COL] == VAL_SET])
    try:
        extract(sin_val, VAL_SET)
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError por conjunto vacio")


def test_test_no_se_carga_por_defecto() -> None:
    assert TEST_SET not in EVAL_SETS
    assert EVAL_SETS == (TRAIN_SET, VAL_SET)


# --- metrics ---


def test_roc_auc_y_pr_auc_de_un_modelo_perfecto() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, score) == 1.0
    assert pr_auc(y, score) == 1.0


def test_roc_auc_de_prediccion_aleatoria_es_0_5() -> None:
    y = np.array([0, 1, 0, 1])
    assert np.isclose(roc_auc(y, np.array([0.4, 0.4, 0.4, 0.4])), 0.5)


def test_conteos_de_matriz_de_confusion() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    counts = confusion_counts(y, pred)
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert sum(counts.values()) == len(y)


def test_summarize_incluye_el_conjunto_minimo_de_specs() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.05, 0.65, 0.02, 0.55])
    out = summarize(y, score, threshold=0.5)
    for key in ("roc_auc", "pr_auc", "recall", "precision", "tn", "fp", "fn", "tp"):
        assert key in out
    assert out["n"] == 8
    assert out["tp"] + out["tn"] + out["fp"] + out["fn"] == 8


def test_metrics_rechazan_una_sola_clase() -> None:
    try:
        summarize(np.zeros(5), np.linspace(0, 1, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError con una sola clase")


# --- folds ---


def test_pliegues_walk_forward_respetan_orden_temporal() -> None:
    dt = pd.Series([86400 + WEEK_SECONDS * (i // 10) for i in range(180)])
    folds = walk_forward_folds(dt, n_folds=3, val_weeks=3)
    assert len(folds) == 3
    wk = week_index(dt)
    for tr, va in folds:
        assert len(tr) > 0 and len(va) > 0
        assert wk.iloc[tr].max() < wk.iloc[va].min()
        assert not set(tr.tolist()) & set(va.tolist())


def test_validacion_de_pliegues_pasa_y_detecta_informacion_futura() -> None:
    dt = pd.Series([86400 + WEEK_SECONDS * (i // 10) for i in range(180)])
    folds = walk_forward_folds(dt)
    assert all(c["pass"] for c in check_folds(dt, folds))
    train_idx, valid_idx = folds[0]
    roto = [(np.concatenate([train_idx, valid_idx]), np.array([], dtype=int))]
    roto.extend(folds[1:])
    assert not all(c["pass"] for c in check_folds(dt, roto))


def test_fold_summary_reporta_tasa_de_fraude_por_pliegue() -> None:
    train, _ = _toy_split_data()
    dt = train.X["TransactionDT"]
    folds = walk_forward_folds(dt)
    summ = fold_summary(dt, train.y, folds)
    assert len(summ) == len(folds)
    assert (summ["n_valid"] > 0).all()
    assert summ["fraude_pct_valid"].notna().all()


def test_pliegues_fallan_con_datos_insuficientes() -> None:
    dt = pd.Series([86400 + WEEK_SECONDS * (i // 10) for i in range(50)])
    try:
        walk_forward_folds(dt, n_folds=3, val_weeks=3)
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError por semanas insuficientes")


# --- modelos y entrenamiento ---


def test_escalera_de_modelos_cubre_las_tres_familias() -> None:
    candidates = build_candidates(seed=42, pos_weight=27.6)
    families = {c.family for c in candidates}
    assert FAMILY_BASELINE in families
    assert FAMILY_CLASSICAL in families
    assert FAMILY_BOOSTING in families
    assert families == {"baseline", "classical", "boosting"}
    assert candidates[0].family == FAMILY_BASELINE
    assert candidates[-1].family == FAMILY_BOOSTING


def test_variante_pos_weight_solo_si_se_proporciona() -> None:
    sin_pes = build_candidates(seed=42)
    con_pes = build_candidates(seed=42, pos_weight=27.6)
    assert not any(c.name == "lgbm_pos_weight" for c in sin_pes)
    assert any(c.name == "lgbm_pos_weight" for c in con_pes)
    assert con_pes[-1].estimator.scale_pos_weight == 27.6


def test_submuestra_es_determinista_y_acotada() -> None:
    train, _ = _toy_split_data()
    a, ya = subsample(train.X, train.y, max_rows=50, seed=42)
    b, yb = subsample(train.X, train.y, max_rows=50, seed=42)
    assert len(a) == 50
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(ya, yb)
    entera, _ = subsample(train.X, train.y, max_rows=None, seed=42)
    assert len(entera) == len(train.X)


def test_desbalance_neg_pos_desde_train() -> None:
    train, _ = _toy_split_data()
    ratio = neg_pos_ratio(train.y)
    assert ratio > 1.0
    assert np.isclose(ratio, (len(train.y) - train.y.sum()) / train.y.sum())


def test_comparacion_es_reproducible() -> None:
    train, valid = _toy_split_data()
    candidatos = [_toy_candidate("dummy_a"), _toy_candidate("dummy_b")]
    tabla_1, _ = run_candidates(candidatos, train, valid, seed=42)
    tabla_2, _ = run_candidates(candidatos, train, valid, seed=42)
    # `fit_s` es tiempo de reloj y por definicion no es reproducible
    metricas = [c for c in tabla_1.columns if c != "fit_s"]
    pd.testing.assert_frame_equal(tabla_1[metricas], tabla_2[metricas])
    assert list(tabla_1.columns) == list(tabla_2.columns)


def test_comparacion_ordena_por_pr_auc() -> None:
    train, valid = _toy_split_data()
    candidatos = [_toy_candidate("dummy_a"), _toy_candidate("dummy_b")]
    tabla, resultados = run_candidates(candidatos, train, valid, seed=42)
    assert tabla["pr_auc"].is_monotonic_decreasing
    assert {r.name for r in resultados} == {"dummy_a", "dummy_b"}
    assert (tabla["n_valid"] == valid.n_rows).all()
