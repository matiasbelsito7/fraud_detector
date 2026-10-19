"""Backfill de los experimentos de las Fases G-J en MLflow.

Los scripts de cada fase ya dejaron su resultado en JSON/CSV dentro de
`reports/` (gitignored) y `docs/`. Re-ejecutarlos para instrumentarlos en vivo
seria costoso (el tuning tarda ~87 min) y en el caso de la Fase I imposible sin
violar el consumo unico de `test` (`constitution.md` Sec. 5). Por eso este modulo
**lee esos artefactos y los registra** como runs de MLflow, sin reentrenar y sin
volver a tocar `test`.

Trazabilidad: cada run guarda `git_commit` de la fase que lo genero,
`split_sha256`, `config_id`/rondas cuando aplica y un tag `test_usado`. La
procedencia exacta queda asi enlazada a codigo, datos, configuracion y
resultados (`specs.md` Sec. 10, `constitution.md` Sec. 9).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.modeling.space import SEARCH_SPACE
from src.tracking.client import Tracker

#: Commit que produjo cada experimento (verificado en el historial del repo).
PHASE_COMMITS = {
    "fase_g": "8ab88dec5876756cca86d50d41bb0e54fb26febd",
    "fase_h": "839825c5b1a385f88a9a8d0125c1835cb72534ad",
    "te02": "95370f89da72d125517d81536fb6e151a09988b0",
    "fase_i": "069adc66113e95f547cbc75ca52bc9ff1e9d68c6",
    "fase_j": "ccaf3b7ffe8c5168905a0d445fb7c5589677abc6",
}

PHASE_LABELS = {
    "fase_g": "G (T-G01)",
    "fase_h": "H (T-H01)",
    "te02": "E (T-E02)",
    "fase_i": "I (T-I01)",
    "fase_j": "J (T-J01)",
}


def _tags(
    phase: str, variant: str, split_sha256: str, test_usado: bool, **extra: Any
) -> dict[str, str]:
    tags = {
        "fase": PHASE_LABELS[phase],
        "variante": variant,
        "git_commit": PHASE_COMMITS[phase],
        "split_sha256": split_sha256,
        "test_usado": str(test_usado).lower(),
    }
    tags.update({k: str(v) for k, v in extra.items()})
    return tags


@dataclass(frozen=True)
class ReportsLayout:
    """Rutas de los artefactos que consume el backfill."""

    root: Path
    modeling: Path
    tuning: Path
    tuning_baseline: Path
    evaluation: Path
    evaluation_baseline: Path
    explainability: Path
    docs: Path
    artifacts: Path

    @classmethod
    def default(cls, root: Path) -> ReportsLayout:
        reports = root / "reports"
        return cls(
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Falta el reporte requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(values: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in values.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _fold_metrics(prefix: str, folds: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in folds:
        fold = int(row.get("fold", 0))
        for key, value in _numeric(row).items():
            if key == "fold":
                continue
            out[f"{prefix}_fold{fold}_{key}"] = value
    return out


def backfill_modeling(tracker: Tracker, layout: ReportsLayout) -> list[str]:
    """Fase G: comparacion de baselines y boosting."""
    data = _read_json(layout.modeling / "comparison.json")
    tags = _tags("fase_g", "fase_g", data["split_sha256"], False)
    parent, created = tracker.ensure_run(
        "fase_g", "G - comparacion de modelos", tags=tags
    )
    tracker.record(parent, created)
    tracker.log_params(
        parent,
        {
            "seed": data["seed"],
            "threshold": data["threshold"],
            "n_features": data["n_features"],
            "pos_weight_train": data["pos_weight_train"],
            "entrenado_en": data["entrenado_en"],
            "evaluado_en": data["evaluado_en"],
        },
    )
    tracker.log_path(parent, layout.modeling / "comparison.csv")
    tracker.log_path(parent, layout.modeling / "comparison.json")
    tracker.log_path(parent, layout.docs / "modeling_report.md")
    tracker.finish(parent)

    run_ids = [parent]
    for model in data["modelos"]:
        key = f"fase_g_{model['nombre']}"
        child, created = tracker.ensure_run(
            key,
            f"G - {model['nombre']}",
            tags={**tags, "familia": model["familia"]},
            parent_run_id=parent,
        )
        tracker.record(child, created)
        tracker.log_params(
            child,
            {
                "familia": model["familia"],
                "n_train": model["n_train"],
                "n_valid": model["n_valid"],
                "nota": model["nota"],
                "seed": data["seed"],
            },
        )
        tracker.log_metrics(
            child,
            {**model["metricas"], "fit_segundos": model["fit_segundos"]},
        )
        tracker.finish(child)
        run_ids.append(child)
    return run_ids


def _backfill_tuning_variant(
    tracker: Tracker,
    search_dir: Path,
    phase: str,
    variant: str,
    title: str,
    model_file: Path | None,
    docs: list[Path],
    extra_params: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
    extra_artifacts: list[Path] | None = None,
) -> list[str]:
    record = _read_json(search_dir / "selected.json")
    selected = record["seleccion"]
    search = _read_json(search_dir / "search.json")
    rows = record.get("busqueda") or search.get("busqueda", [])
    tags = _tags(phase, variant, record["split_sha256"], False)

    parent, created = tracker.ensure_run(f"{variant}_seleccion", title, tags=tags)
    tracker.record(parent, created)
    params: dict[str, Any] = {
        "seed": record["seed"],
        "n_configs": record["n_configs"],
        "n_features": record["n_features"],
        "folds": record["folds"],
        "criterio_seleccion": record["criterio_seleccion"],
        "desempate": record["desempate"],
        "config_id": selected["config_id"],
        "best_iteration": selected["best_iteration"],
        "is_anchor": selected["is_anchor"],
        **{f"params_{k}": v for k, v in selected["params"].items()},
        **(extra_params or {}),
    }
    tracker.log_params(parent, params)
    tracker.log_metrics(
        parent,
        {
            "seleccion_mean_pr_auc": selected["mean_pr_auc"],
            "seleccion_mean_roc_auc": selected["mean_roc_auc"],
            "seleccion_mean_recall": selected["mean_recall"],
            **{f"validation_{k}": v for k, v in _numeric(record["validacion"]).items()},
            **(extra_metrics or {}),
        },
    )
    tracker.log_path(parent, search_dir / "selected.json")
    tracker.log_path(parent, search_dir / "search.json")
    tracker.log_path(parent, search_dir / "search.csv")
    for path in docs:
        tracker.log_path(parent, path)
    for path in extra_artifacts or []:
        tracker.log_path(parent, path)
    if model_file is not None:
        tracker.log_model(parent, model_file)
    tracker.finish(parent)

    run_ids = [parent]
    for row in rows:
        config_id = int(row["config_id"])
        key = f"{variant}_config_{config_id}"
        child, created = tracker.ensure_run(
            key,
            f"{title} - config {config_id}",
            tags={
                **tags,
                "config_id": str(config_id),
                "is_anchor": str(row["is_anchor"]).lower(),
            },
            parent_run_id=parent,
        )
        tracker.record(child, created)
        tracker.log_params(
            child,
            {
                "config_id": config_id,
                "is_anchor": row["is_anchor"],
                **{k: row[k] for k in SEARCH_SPACE if k in row},
            },
        )
        tracker.log_metrics(
            child,
            {
                "mean_pr_auc": row["mean_pr_auc"],
                "std_pr_auc": row["std_pr_auc"],
                "min_pr_auc": row["min_pr_auc"],
                "mean_roc_auc": row["mean_roc_auc"],
                "mean_recall": row["mean_recall"],
                "best_iteration": row["best_iteration"],
                "seconds": row["seconds"],
                **_fold_metrics("search", row.get("folds", [])),
            },
        )
        tracker.finish(child)
        run_ids.append(child)
    return run_ids


def backfill_tuning_fase_h(tracker: Tracker, layout: ReportsLayout) -> list[str]:
    """Fase H: busqueda y seleccion sin features de cliente (762 features)."""
    return _backfill_tuning_variant(
        tracker,
        search_dir=layout.tuning_baseline,
        phase="fase_h",
        variant="fase_h",
        title="H - tuning y seleccion",
        model_file=layout.artifacts / "model_final_fase_h.joblib",
        docs=[layout.docs / "tuning_report.md"],
    )


def backfill_tuning_te02(tracker: Tracker, layout: ReportsLayout) -> list[str]:
    """T-E02: busqueda y seleccion con identidad de cliente (791 features)."""
    threshold = _read_json(layout.evaluation / "threshold.json")
    return _backfill_tuning_variant(
        tracker,
        search_dir=layout.tuning,
        phase="te02",
        variant="te02",
        title="E02 - tuning con identidad de cliente",
        model_file=layout.artifacts / "model_final.joblib",
        docs=[layout.docs / "client_features_report.md"],
        extra_params={
            "umbral_operativo": threshold["threshold"],
            "target_recall": threshold["target_recall"],
            "oof_coverage": threshold["oof_coverage"],
        },
        extra_metrics={
            **{
                f"umbral_validacion_{k}": v
                for k, v in _numeric(threshold["calibracion_validation"]).items()
            },
            **{
                f"umbral_oof_{k}": v
                for k, v in _numeric(threshold["corroboracion_oof_train"]).items()
            },
            "deriva_recall_oof_menos_validation": threshold[
                "deriva_recall_oof_menos_validation"
            ],
        },
        extra_artifacts=[layout.evaluation / "threshold.json"],
    )


def backfill_evaluation(tracker: Tracker, layout: ReportsLayout) -> list[str]:
    """Fase I: unico consumo de `test`, con el modelo de Fase H."""
    data = _read_json(layout.evaluation_baseline / "test_metrics.json")
    metrics = data["metricas_test"]
    tags = _tags(
        "fase_i",
        "fase_h",
        data["split_sha256"],
        True,
        advertencia=(
            "test consumido una unica vez; el modelo es Fase H sin features de "
            "identidad de cliente"
        ),
    )
    run, created = tracker.ensure_run(
        "fase_i_test", "I - evaluacion final en test", tags=tags
    )
    tracker.record(run, created)
    tracker.log_params(
        run,
        {
            "conjunto": data["conjunto"],
            "umbral": data["umbral"],
            "config_id": data["config_id"],
            "rounds": data["rounds"],
            "criterio": data["criterio"],
            "origen_del_umbral": data["origen_del_umbral"],
            "umbral_fijado_antes_de_test": data["umbral_fijado_antes_de_test"],
        },
    )
    numeric = _numeric(metrics)
    numeric.update(
        {f"recall_ci_95_{i}": v for i, v in enumerate(metrics.get("recall_ci_95", []))}
    )
    numeric["cumple_objetivo_recall"] = float(bool(metrics["cumple_objetivo_recall"]))
    tracker.log_metrics(run, numeric)
    tracker.log_path(run, layout.evaluation_baseline / "test_metrics.json")
    tracker.log_path(run, layout.evaluation_baseline / "test_sweep.csv")
    tracker.log_path(run, layout.evaluation_baseline / "test_roc_curve.csv")
    tracker.log_path(run, layout.evaluation_baseline / "test_pr_curve.csv")
    tracker.log_path(run, layout.evaluation_baseline / "threshold.json")
    tracker.log_path(run, layout.docs / "evaluation_report.md")
    tracker.finish(run)
    return [run]


def backfill_explainability(tracker: Tracker, layout: ReportsLayout) -> list[str]:
    """Fase J: importancia global y SHAP sobre validation, sin test."""
    data = _read_json(layout.explainability / "summary.json")
    model = data["modelo"]
    tags = _tags("fase_j", "te02", data["split_sha256"], False)
    run, created = tracker.ensure_run(
        "fase_j_explainability", "J - explainability", tags=tags
    )
    tracker.record(run, created)
    tracker.log_params(
        run,
        {
            "conjunto_explicado": data["conjunto_explicado"],
            "modelo_artefacto": model["artefacto"],
            "modelo_clase": model["clase"],
            "modelo_config_id": model["config_id"],
            "modelo_n_rondas": model["n_rondas"],
            "modelo_n_features": model["n_features"],
            "shap_tecnica": data["shap"]["tecnica"],
            "muestra_n": data["muestra_shap"]["n"],
            "muestra_seed": data["muestra_shap"]["seed"],
            "modelo_pendiente_de_decision": data["modelo_pendiente_de_decision"],
        },
    )
    tracker.log_metrics(
        run,
        {
            "umbral_operativo": data["umbral_operativo"],
            "shap_base_value_medio": data["shap"]["base_value_medio"],
            "shap_error_aditividad_max": data["shap"]["error_aditividad_max"],
        },
    )
    for name in (
        "summary.json",
        "importance_gain.csv",
        "importance_gain_by_family.csv",
        "shap_global.csv",
        "shap_by_family.csv",
    ):
        tracker.log_path(run, layout.explainability / name)
    tracker.log_path(run, layout.docs / "explainability_report.md")
    tracker.finish(run)
    return [run]


BACKFILLS = {
    "g": ("Fase G - modelado", backfill_modeling),
    "h": ("Fase H - tuning (sin cliente)", backfill_tuning_fase_h),
    "e02": ("T-E02 - tuning (con cliente)", backfill_tuning_te02),
    "i": ("Fase I - evaluacion en test", backfill_evaluation),
    "j": ("Fase J - explainability", backfill_explainability),
}


def run_backfill(
    tracker: Tracker,
    layout: ReportsLayout,
    only: list[str] | None = None,
) -> dict[str, list[str]]:
    """Ejecuta el backfill de las fases indicadas (todas por defecto)."""
    selected = only or list(BACKFILLS)
    results: dict[str, list[str]] = {}
    for key in selected:
        if key not in BACKFILLS:
            raise ValueError(f"Fase desconocida: {key}. Opciones: {list(BACKFILLS)}")
        _label, function = BACKFILLS[key]
        results[key] = function(tracker, layout)
    return results
