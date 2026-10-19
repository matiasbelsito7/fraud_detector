from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.explainability.families import FAMILIES, feature_family
from src.explainability.importance import (
    aggregate_by_family,
    check_additivity,
    gain_importance,
    local_contributions,
    shap_base_value,
    shap_contributions,
    shap_summary,
)


def _toy_model() -> tuple[LGBMClassifier, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 400
    X = pd.DataFrame(
        {
            "V1": rng.normal(0, 1, n),
            "C1": rng.normal(0, 1, n),
            "card1": rng.integers(0, 8, n).astype(float),
            "hour_sin": rng.normal(0, 1, n),
            "ruido": rng.normal(0, 1, n),
        }
    )
    y = ((X["V1"] + 0.5 * X["C1"] + rng.normal(0, 0.3, n)) > 0.6).astype(int)
    model = LGBMClassifier(
        n_estimators=8,
        num_leaves=3,
        min_child_samples=5,
        objective="binary",
        random_state=42,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)
    return model, X


def test_family_cubre_las_familias_del_dataset() -> None:
    assert feature_family("V12") == "V (Vesta)"
    assert feature_family("C3") == "C (contadores)"
    assert feature_family("D1") == "D (tiempo)"
    assert feature_family("M4") == "M (match)"
    assert feature_family("card1") == "card (tarjeta)"
    assert feature_family("addr2") == "addr (direccion)"
    assert feature_family("dist1") == "dist (distancia)"
    assert feature_family("P_emaildomain") == "email (dominio)"
    assert feature_family("id_31") == "id_ (identity)"
    assert feature_family("missingindicator_V1") == "indicador de missing"
    assert feature_family("hour_cos") == "FE: tiempo"
    assert feature_family("cnt_card1") == "FE: frecuencia"
    assert feature_family("amt_rel_card1") == "FE: importes"
    assert feature_family("gap_sec_card1") == "FE: historial tarjeta"
    assert feature_family("p_eq_r_email") == "FE: entidades"
    assert feature_family("mean_C1_prev_client") == "FE: identidad de cliente"
    assert feature_family("mean_C14_prev_uid") == "FE: identidad de cliente"
    assert feature_family("cnt_client_uid") == "FE: identidad de cliente"
    assert feature_family("TransactionAmt") == "otras"
    assert feature_family("V12") in FAMILIES


def test_gain_importance_ordena_y_reparte_el_100_por_ciento() -> None:
    model, X = _toy_model()
    frame = gain_importance(model, list(X.columns))
    assert list(frame["rank"]) == list(range(1, len(X.columns) + 1))
    assert frame["gain"].is_monotonic_decreasing
    assert frame["gain"].sum() > 0
    assert frame["gain_pct"].sum() == pytest.approx(100.0)
    assert frame["split"].sum() > 0
    assert set(frame["family"]) >= {"V (Vesta)", "C (contadores)", "card (tarjeta)"}


def test_gain_rechaza_objetos_que_no_son_booster() -> None:
    with pytest.raises(TypeError, match="booster_"):
        gain_importance(object(), ["a"])


def test_gain_valida_el_numero_de_nombres() -> None:
    model, X = _toy_model()
    with pytest.raises(ValueError, match="nombres"):
        gain_importance(model, ["solo_una"])


def test_shap_reconstruye_el_score_crudo() -> None:
    """Invariante central: bias + suma(contribuciones) == raw_score."""
    model, X = _toy_model()
    assert check_additivity(model, X) < 1e-6


def test_shap_tiene_una_columna_de_bias_y_rechaza_formas_invalidas() -> None:
    model, X = _toy_model()
    contributions = shap_contributions(model, X)
    assert contributions.shape == (len(X), len(X.columns) + 1)
    assert np.isfinite(contributions).all()
    assert np.isfinite(shap_base_value(contributions))

    class _Plano:
        def predict(self, X, pred_contrib=False):
            return np.zeros(5)

    with pytest.raises(ValueError, match="forma inesperada"):
        shap_contributions(_Plano(), X)


def test_shap_summary_mide_la_media_absoluta_por_feature() -> None:
    model, X = _toy_model()
    contributions = shap_contributions(model, X)[:, :-1]
    frame = shap_summary(model, X, list(X.columns))
    row = frame.set_index("feature").loc["V1"]
    assert row["mean_abs_shap"] == pytest.approx(np.abs(contributions[:, 0]).mean())
    assert row["mean_shap"] == pytest.approx(contributions[:, 0].mean())
    assert frame["mean_abs_shap"].is_monotonic_decreasing
    assert frame.loc[frame["feature"] == "card1", "family"].iloc[0] == "card (tarjeta)"


def test_shap_summary_valida_el_numero_de_nombres() -> None:
    model, X = _toy_model()
    with pytest.raises(ValueError, match="nombres"):
        shap_summary(model, X, list(X.columns)[:-1])


def test_local_contributions_devuelve_solo_las_top_y_con_signo() -> None:
    model, X = _toy_model()
    explanation = local_contributions(model, X.iloc[[0]], list(X.columns), top=3)
    assert len(explanation["top"]) == 3
    ordered = [abs(item["shap"]) for item in explanation["top"]]
    assert ordered == sorted(ordered, reverse=True)
    assert explanation["raw_score"] == pytest.approx(
        explanation["base_value"] + sum(c for c in _all(model, X.iloc[[0]]))
    )


def _all(model: LGBMClassifier, row: pd.DataFrame) -> np.ndarray:
    return shap_contributions(model, row)[0, :-1]


def test_local_contributions_valida_la_forma_de_la_fila() -> None:
    model, X = _toy_model()
    with pytest.raises(ValueError, match="una fila"):
        local_contributions(model, X.iloc[:2], list(X.columns))
    with pytest.raises(ValueError, match="columnas"):
        local_contributions(
            model, X[list(reversed(X.columns))].iloc[[0]], list(X.columns)
        )


def test_aggregate_by_family_conserva_el_total() -> None:
    model, X = _toy_model()
    frame = gain_importance(model, list(X.columns))
    by_family = aggregate_by_family(frame, "gain_pct")
    assert by_family["gain_pct"].sum() == pytest.approx(100.0)
    assert by_family["n_features"].sum() == len(X.columns)
    assert by_family["gain_pct"].is_monotonic_decreasing
    assert by_family["pct"].sum() == pytest.approx(100.0)


def test_aggregate_by_family_exige_la_columna_family() -> None:
    with pytest.raises(ValueError, match="family"):
        aggregate_by_family(pd.DataFrame({"feature": ["a"], "x": [1.0]}), "x")


def test_el_muestreo_del_runner_es_determinista_y_ordenado() -> None:
    from run_explainability import _sample_indices

    a = _sample_indices(1000, 100, seed=42)
    b = _sample_indices(1000, 100, seed=42)
    c = _sample_indices(1000, 100, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.size == 100
    assert np.all(np.diff(a) > 0)
    assert np.array_equal(_sample_indices(50, 100, seed=1), np.arange(50))


def test_el_runner_no_carga_test_y_muestrea_validation() -> None:
    import inspect

    from run_explainability import main

    src = inspect.getsource(main)
    assert "TEST_SET not in splits" in src
    assert "names=(VAL_SET,)" in src
    assert "valid = splits[VAL_SET]" in src
