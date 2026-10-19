from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pytest
from lightgbm import LGBMClassifier

from src.tracking.backfill import (
    ReportsLayout,
    backfill_evaluation,
    backfill_explainability,
    backfill_modeling,
    backfill_tuning_fase_h,
    backfill_tuning_te02,
    run_backfill,
)
from src.tracking.client import Tracker, configure_tracking

SPLIT_SHA = "a" * 64
OLD_SHA = "b" * 64


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metrics(prefix: float = 0.0) -> dict[str, float]:
    return {
        "n": 50,
        "pos_rate": 0.04,
        "threshold": 0.5,
        "roc_auc": 0.9 + prefix,
        "pr_auc": 0.6 + prefix,
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.44,
        "predict_rate": 0.05,
        "tn": 45,
        "fp": 3,
        "fn": 1,
        "tp": 1,
    }


def _search_row(config_id: int, anchor: bool) -> dict[str, Any]:
    return {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": 1.0,
        "mean_pr_auc": 0.6 + config_id * 0.001,
        "std_pr_auc": 0.05,
        "min_pr_auc": 0.55,
        "mean_roc_auc": 0.92,
        "mean_recall": 0.42,
        "best_iteration": 200,
        "seconds": 12.0,
        "folds": [
            {
                "fold": 0,
                "n_train": 100,
                "n_valid": 30,
                "best_iteration": 200,
                "pr_auc": 0.61,
                "roc_auc": 0.92,
                "recall": 0.4,
                "precision": 0.8,
            },
        ],
        "config_id": config_id,
        "is_anchor": anchor,
    }


def _selected(search_dir: Path, config_id: int, n_features: int, sha: str) -> None:
    selected = {
        "seed": 42,
        "n_configs": 2,
        "split_file": "split.parquet",
        "split_sha256": sha,
        "n_features": n_features,
        "folds": 3,
        "criterio_seleccion": "PR-AUC medio",
        "desempate": "ROC-AUC medio",
        "test_usado": False,
        "entrenado_en": "train",
        "confirmado_en": "validation",
        "seleccion": {
            "params": _search_row(config_id, False),
            "best_iteration": 200,
            "mean_pr_auc": 0.6,
            "mean_roc_auc": 0.92,
            "mean_recall": 0.42,
            "config_id": config_id,
            "is_anchor": False,
        },
        "validacion": _metrics(),
        "modelo": {
            "artefacto": "model.joblib",
            "clase": "LGBMClassifier",
            "n_rondas": 200,
        },
        "busqueda": [_search_row(0, True), _search_row(1, False)],
    }
    _write_json(search_dir / "selected.json", selected)
    _write_json(search_dir / "search.json", {**selected, "modo": "busqueda"})


def _model(path: Path) -> Path:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = (X["a"] + rng.normal(scale=0.1, size=40) > 0).astype(int)
    model = LGBMClassifier(n_estimators=3, verbose=-1)
    model.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


@pytest.fixture()
def layout(tmp_path: Path) -> ReportsLayout:
    root = tmp_path
    reports = root / "reports"
    base = ReportsLayout(
        root=root,
        modeling=reports / "modeling",
        tuning=reports / "tuning",
        tuning_baseline=reports / "tuning" / "baseline_fase_h",
        evaluation=reports / "evaluation",
        evaluation_baseline=reports / "evaluation" / "fase_i_baseline",
        explainability=reports / "explainability",
        docs=root / "docs",
        artifacts=root / "artifacts",
    )

    _write_json(
        base.modeling / "comparison.json",
        {
            "seed": 42,
            "threshold": 0.5,
            "entrenado_en": "train",
            "evaluado_en": "validation",
            "test_usado": False,
            "modelos": [
                {
                    "nombre": "lgbm",
                    "familia": "boosting",
                    "nota": "referencia",
                    "n_train": 100,
                    "n_valid": 50,
                    "fit_segundos": 1.5,
                    "metricas": _metrics(),
                }
            ],
            "split_file": "split.parquet",
            "split_sha256": SPLIT_SHA,
            "n_features": 10,
            "pos_weight_train": 27.5,
            "folds": [{"fold": 0}],
        },
    )
    _touch(base.modeling / "comparison.csv")
    _touch(base.docs / "modeling_report.md")

    _selected(base.tuning_baseline, config_id=1, n_features=10, sha=SPLIT_SHA)
    _touch(base.tuning_baseline / "search.csv")
    _touch(base.docs / "tuning_report.md")

    _selected(base.tuning, config_id=1, n_features=12, sha=SPLIT_SHA)
    _touch(base.tuning / "search.csv")
    _touch(base.docs / "client_features_report.md")
    _write_json(
        base.evaluation / "threshold.json",
        {
            "threshold": 0.008,
            "target_recall": 0.8,
            "oof_coverage": 0.46,
            "calibracion_validation": _metrics(),
            "corroboracion_oof_train": _metrics(),
            "deriva_recall_oof_menos_validation": -0.01,
        },
    )

    _write_json(
        base.evaluation_baseline / "test_metrics.json",
        {
            "conjunto": "test",
            "umbral": 0.029,
            "criterio": "recall objetivo >= 0.8",
            "config_id": 18,
            "rounds": 219,
            "split_sha256": OLD_SHA,
            "origen_del_umbral": "validation",
            "umbral_fijado_antes_de_test": True,
            "metricas_test": {
                **_metrics(),
                "recall_ci_95": [0.75, 0.78],
                "cumple_objetivo_recall": False,
            },
            "referencia_validacion": _metrics(),
            "referencia_oof_train": _metrics(),
        },
    )
    for name in (
        "test_sweep.csv",
        "test_roc_curve.csv",
        "test_pr_curve.csv",
        "threshold.json",
    ):
        _touch(base.evaluation_baseline / name)
    _touch(base.docs / "evaluation_report.md")

    _write_json(
        base.explainability / "summary.json",
        {
            "fase": "J (T-J01)",
            "conjunto_explicado": "validation",
            "test_usado": False,
            "modelo": {
                "artefacto": "model_final.joblib",
                "clase": "LGBMClassifier",
                "config_id": 11,
                "n_rondas": 439,
                "n_features": 791,
            },
            "split_sha256": SPLIT_SHA,
            "umbral_operativo": 0.008,
            "muestra_shap": {"n": 5000, "seed": 42, "metodo": "uniforme"},
            "shap": {
                "tecnica": "TreeSHAP nativo",
                "base_value_medio": -6.9,
                "error_aditividad_max": 2.2e-08,
            },
            "modelo_pendiente_de_decision": True,
        },
    )
    for name in (
        "importance_gain.csv",
        "importance_gain_by_family.csv",
        "shap_global.csv",
        "shap_by_family.csv",
    ):
        _touch(base.explainability / name)
    _touch(base.docs / "explainability_report.md")

    _model(base.artifacts / "model_final.joblib")
    _model(base.artifacts / "model_final_fase_h.joblib")
    return base


@pytest.fixture()
def tracker(tmp_path: Path) -> Tracker:
    client, experiment_id = configure_tracking(
        "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix(), "test_exp"
    )
    return Tracker(client=client, experiment_id=experiment_id, root=tmp_path)


def _runs(tracker: Tracker) -> list[Any]:
    return list(tracker.client.search_runs([tracker.experiment_id]))


def _keys(tracker: Tracker) -> list[str]:
    return [run.data.tags.get("tracking_key", "") for run in _runs(tracker)]


def test_backfill_modeling_crea_padre_e_hijos(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    ids = backfill_modeling(tracker, layout)
    assert len(ids) == 2
    assert len(_runs(tracker)) == 2
    parent = tracker.client.get_run(ids[0])
    assert parent.data.tags["fase"] == "G (T-G01)"
    assert parent.data.tags["test_usado"] == "false"
    assert parent.data.params["n_features"] == "10"
    child = tracker.client.get_run(ids[1])
    assert child.data.tags["familia"] == "boosting"
    assert child.data.metrics["pr_auc"] == pytest.approx(0.6)


def test_backfill_tuning_crea_config_runs_y_modelo(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    ids = backfill_tuning_fase_h(tracker, layout)
    assert len(ids) == 3
    parent = tracker.client.get_run(ids[0])
    assert parent.data.tags["variante"] == "fase_h"
    assert parent.data.params["config_id"] == "1"
    model_artifacts = tracker.client.list_artifacts(ids[0], path="model")
    assert any(item.path.endswith("MLmodel") for item in model_artifacts)
    child = tracker.client.get_run(ids[1])
    assert "search_fold0_pr_auc" in child.data.metrics
    assert "num_leaves" in child.data.params


def test_backfill_tuning_te02_registra_umbral_y_cliente(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    ids = backfill_tuning_te02(tracker, layout)
    parent = tracker.client.get_run(ids[0])
    assert parent.data.tags["variante"] == "te02"
    assert parent.data.params["umbral_operativo"] == "0.008"
    assert "umbral_validacion_pr_auc" in parent.data.metrics


def test_backfill_evaluation_marca_test_consumido(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    ids = backfill_evaluation(tracker, layout)
    run = tracker.client.get_run(ids[0])
    assert run.data.tags["test_usado"] == "true"
    assert run.data.tags["variante"] == "fase_h"
    assert run.data.metrics["recall"] == pytest.approx(0.4)
    assert "advertencia" in run.data.tags


def test_backfill_explainability_registra_shap(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    ids = backfill_explainability(tracker, layout)
    run = tracker.client.get_run(ids[0])
    assert run.data.tags["variante"] == "te02"
    assert run.data.metrics["shap_error_aditividad_max"] == pytest.approx(2.2e-08)
    assert run.data.params["modelo_config_id"] == "11"


def test_backfill_es_idempotente_sin_force(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    first = run_backfill(tracker, layout)
    assert tracker.created
    created_after_first = len(tracker.created)
    tracker.created.clear()
    tracker.skipped.clear()
    second = run_backfill(tracker, layout)
    assert not tracker.created
    assert tracker.skipped
    assert sum(len(v) for v in first.values()) == sum(len(v) for v in second.values())
    assert created_after_first > 0


def test_backfill_force_duplica(layout: ReportsLayout, tracker: Tracker) -> None:
    run_backfill(tracker, layout, only=["g"])
    tracker.created.clear()
    tracker.force = True
    run_backfill(tracker, layout, only=["g"])
    assert tracker.created
    assert _keys(tracker).count("fase_g") == 2


def test_run_backfill_rechaza_fase_desconocida(
    layout: ReportsLayout, tracker: Tracker
) -> None:
    with pytest.raises(ValueError):
        run_backfill(tracker, layout, only=["z"])


def test_no_carga_test() -> None:
    source = (Path(__file__).resolve().parent.parent / "src" / "tracking").glob("*.py")
    text = "\n".join(path.read_text(encoding="utf-8") for path in source)
    assert "load_splits" not in text
    assert "test_transaction.csv" not in text
