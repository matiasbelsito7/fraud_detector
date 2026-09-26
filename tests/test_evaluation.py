from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.dataset import SplitData, extract
from src.modeling.evaluate import (
    TARGET_RECALL,
    curve_points,
    out_of_fold_scores,
    recall_interval,
    threshold_for_target_recall,
    threshold_sweep,
)
from src.modeling.folds import walk_forward_folds
from src.modeling.metrics import summarize
from src.modeling.space import BASE_PARAMS


def _toy_split(n_weeks: int = 12, rows_per_week: int = 50, seed: int = 11) -> SplitData:
    """Serie temporal pequena con senal aprendible en `f0`.

    El fraude depende de `f0` con un poco de ruido, de modo que un modelo
    entrenado sin ver una fila esa fila la ordena bien: eso hace significativo
    comprobar que las predicciones out-of-fold no son producto de memorizacion.
    """
    rng = np.random.default_rng(seed)
    n = n_weeks * rows_per_week
    dt = np.repeat(np.arange(n_weeks) * 604_800, rows_per_week)
    f0 = rng.normal(0, 1, n).astype("float32")
    f1 = rng.normal(0, 1, n).astype("float32")
    X = pd.DataFrame({"TransactionDT": dt, "f0": f0, "f1": f1})
    y = pd.Series((f0 + rng.normal(0, 0.5, n) > 0.7).astype("int8"))
    return SplitData(name="train", X=X, y=y)


def _fast_params() -> dict[str, float]:
    return {**BASE_PARAMS, "learning_rate": 0.3, "scale_pos_weight": 1.0}


def test_el_umbral_alcanza_exactamente_el_recall_objetivo() -> None:
    # 10 positivos con puntuaciones 0.9..0.0 y 90 negativos por debajo de 0.05
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] + [0.05] * 90)
    y = np.array([1] * 10 + [0] * 90)
    t = threshold_for_target_recall(y, s, 0.8)
    m = summarize(y, s, threshold=t)
    assert m["recall"] == pytest.approx(0.8)
    assert m["tp"] == 8


def test_el_umbral_es_el_mas_bajo_que_alcanza_el_objetivo() -> None:
    """El corte debe ser ajustado: subirlo un ulp pierde el recall objetivo."""
    rng = np.random.default_rng(5)
    y = (rng.random(4000) < 0.05).astype(int)
    s = rng.random(4000)
    t = threshold_for_target_recall(y, s, 0.8)
    at = summarize(y, s, threshold=t)
    below = summarize(y, s, threshold=float(np.nextafter(t, 0.0)))
    above = summarize(y, s, threshold=float(np.nextafter(t, 1.0)))
    assert at["recall"] >= 0.8
    assert above["recall"] < 0.8
    # bajar el umbral nunca reduce el recall, y aqui no cambia nada
    assert below["recall"] == pytest.approx(at["recall"])


def test_mas_recall_objetivo_nunca_baja_el_umbral() -> None:
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.08).astype(int)
    s = rng.random(3000)
    thresholds = [threshold_for_target_recall(y, s, r) for r in (0.5, 0.8, 0.95)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_el_umbral_rechaza_objetivos_imposibles_o_sin_positivos() -> None:
    y = np.array([1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="target_recall"):
        threshold_for_target_recall(y, s, 0.0)
    with pytest.raises(ValueError, match="Sin positivos"):
        threshold_for_target_recall(np.zeros(4), s, 0.8)


def test_el_umbral_con_empates_no_degrada_el_recall() -> None:
    y = np.array([1] * 10 + [0] * 10)
    s = np.array([0.5] * 20)
    m = summarize(y, s, threshold=threshold_for_target_recall(y, s, 0.8))
    # con empate total en la puntuacion el umbral arrastra a todos los positivos
    assert m["recall"] == pytest.approx(1.0)


def test_el_objetivo_por_defecto_es_el_negocio_acordado() -> None:
    assert TARGET_RECALL == 0.80


def test_el_barrido_es_monotono_en_recall_y_recubre_el_rango() -> None:
    """Recordar: recall DECRECE al subir el umbral. predict_rate tambien."""
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.05).astype(int)
    s = rng.random(2000)
    sweep = threshold_sweep(y, s, n_points=51)
    assert len(sweep) == 51
    assert sweep["threshold"].is_monotonic_increasing
    assert np.all(np.diff(sweep["recall"].to_numpy()) <= 1e-12)
    assert np.all(np.diff(sweep["predict_rate"].to_numpy()) <= 1e-12)
    assert sweep["recall"].iloc[0] == pytest.approx(1.0)
    assert sweep["predict_rate"].iloc[0] == pytest.approx(1.0)
    assert sweep["predict_rate"].iloc[-1] == pytest.approx(0.0)


def test_las_predicciones_out_of_fold_no_ven_su_propia_etiqueta() -> None:
    """Una fila puntuada por un modelo que la entreno daria recall ~1.

    Se comprueba que el score ordena bien fuera de muestra (PR-AUC alta) sin
    identificar la etiqueta (recall@0,5 lejos de 1): generaliza, no memoriza.
    """
    split = _toy_split(n_weeks=16, rows_per_week=60, seed=13)
    folds = walk_forward_folds(split.X["TransactionDT"], n_folds=3, val_weeks=1)
    raw = out_of_fold_scores(_fast_params(), 60, split, folds, seed=42)["scores"]
    mask = np.isfinite(raw)
    y = np.asarray(split.y).astype(int)[mask]
    s = raw[mask]
    assert mask.sum() == 180
    assert np.unique(y).size == 2
    # muy por encima de la tasa base: el modelo ordena fuera de muestra
    assert summarize(y, s)["pr_auc"] > 2.0 * y.mean()
    assert summarize(y, s, threshold=0.5)["recall"] < 0.99


def test_las_predicciones_out_of_fold_dejan_sin_puntuar_la_ventana_inicial() -> None:
    """Hallazgo de la Fase I: con ventana expansiva las primeras semanas solo
    entrenan, asi que no se puntuan. La cobertura se reporta, no se oculta."""
    split = _toy_split(n_weeks=10, rows_per_week=10)
    folds = walk_forward_folds(split.X["TransactionDT"], n_folds=2, val_weeks=2)
    out = out_of_fold_scores(_fast_params(), 5, split, folds, seed=42)
    scored = out["n_scored"]
    assert scored == 40  # semanas 6-9
    assert out["n_unscored"] == 60  # semanas 0-5, solo entrenamiento
    assert out["coverage"] == pytest.approx(0.4)
    assert np.isnan(out["scores"][:60]).all()
    assert np.isfinite(out["scores"][60:]).all()


def test_strict_exige_cobertura_total() -> None:
    split = _toy_split(n_weeks=10, rows_per_week=10)
    folds = walk_forward_folds(split.X["TransactionDT"], n_folds=2, val_weeks=2)
    with pytest.raises(ValueError, match="sin puntuar"):
        out_of_fold_scores(_fast_params(), 5, split, folds, seed=42, strict=True)


def test_las_predicciones_out_of_fold_fallan_si_un_pliegue_esta_vacio() -> None:
    split = _toy_split(n_weeks=10, rows_per_week=10)
    folds = walk_forward_folds(split.X["TransactionDT"], n_folds=2, val_weeks=2)
    partial = [(folds[0][0], folds[0][1][:0])]
    with pytest.raises(ValueError, match="2 dimensional"):
        out_of_fold_scores(_fast_params(), 5, split, partial, seed=42)


def test_el_intervalo_de_recall_se_estrecha_con_mas_positivos() -> None:
    lo_small, hi_small = recall_interval(0.8, 200)
    lo_big, hi_big = recall_interval(0.8, 20000)
    assert (hi_small - lo_small) > (hi_big - lo_big)
    assert 0.0 <= lo_big <= 0.8 <= hi_big <= 1.0
    assert np.isnan(recall_interval(0.8, 0)[0])


def test_la_evaluacion_no_puede_alcanzar_validation_ni_test() -> None:
    """`out_of_fold_scores` no tiene forma de recibir validation ni test."""
    import inspect

    sig = inspect.signature(out_of_fold_scores)
    assert "validation" not in sig.parameters
    assert "test" not in sig.parameters
    assert list(sig.parameters) == [
        "params",
        "rounds",
        "train",
        "folds",
        "seed",
        "strict",
    ]
    # y el umbral tampoco acepta otro conjunto: solo y_true/y_score ya puntuados
    for fn in (threshold_for_target_recall, threshold_sweep):
        assert "validation" not in inspect.signature(fn).parameters
        assert "test" not in inspect.signature(fn).parameters


def test_el_barrido_reproduce_las_metricas_de_summarize() -> None:
    rng = np.random.default_rng(9)
    y = (rng.random(1500) < 0.2).astype(int)
    s = rng.random(1500)
    sweep = threshold_sweep(y, s, n_points=21)
    row = sweep[np.isclose(sweep["threshold"].to_numpy(), 0.5)].iloc[0]
    ref = summarize(y, s, threshold=0.5)
    assert row["recall"] == pytest.approx(ref["recall"])
    assert row["precision"] == pytest.approx(ref["precision"])
    assert row["pr_auc"] == pytest.approx(ref["pr_auc"])
    assert row["fp"] == ref["fp"]


def test_las_curvas_roc_y_pr_son_alineadas() -> None:
    """precision_recall_curve anade un punto sin threshold: hay que recortarlo.

    Sin esto el DataFrame de la curva PR no se puede construir.
    """
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.08).astype(int)
    s = rng.random(3000)
    c = curve_points(y, s)
    assert c["fpr"].shape == c["tpr"].shape == c["roc_threshold"].shape
    assert c["precision"].shape == c["recall"].shape == c["pr_threshold"].shape
    pd.DataFrame({"fpr": c["fpr"], "tpr": c["tpr"], "threshold": c["roc_threshold"]})
    pd.DataFrame(
        {
            "recall": c["recall"],
            "precision": c["precision"],
            "threshold": c["pr_threshold"],
        }
    )


def test_run_threshold_no_carga_test_y_calibra_en_validation() -> None:
    """Invariante de la Fase I, en el codigo que fija el umbral.

    El umbral debe salir de `validation` (unico bloque puntuado por el modelo
    final) y el script no debe tener forma de llegar a `test`.
    """
    import inspect

    from run_threshold import main

    src = inspect.getsource(main)
    assert "TEST_SET not in splits" in src
    assert "names=EVAL_SETS" in src
    assert "load_splits" in src
    # el threshold se calcula sobre validation, no sobre las variables OOF
    line = [
        ln for ln in src.splitlines() if "threshold = threshold_for_target_recall" in ln
    ]
    assert len(line) == 1
    assert "y_valid" in line[0] and "y_oof" not in line[0]


def test_extract_sigue_excluyendo_test_por_default() -> None:
    df = pd.DataFrame(
        {
            "TransactionID": range(6),
            "TransactionDT": range(6),
            "isFraud": [0, 1, 0, 0, 1, 0],
            "split_set": ["train", "train", "validation", "test", "test", "test"],
            "f": np.arange(6, dtype="float32"),
        }
    )
    out = extract(df, "train")
    assert out.n_rows == 2
    assert "f" in out.X.columns
    assert "isFraud" not in out.X.columns
    assert "split_set" not in out.X.columns
