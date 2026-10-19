"""Cliente y utilidades de tracking para MLflow.

El store por defecto es SQLite bajo `mlruns/` (gitignored). MLflow 3.x puso el
backend de sistema de archivos en modo mantenimiento y recomienda una base de
datos; SQLite no requiere servicio, mantiene todo en un solo archivo y sigue
siendo local. El backend de archivos se evita a proposito para no depender de
`MLFLOW_ALLOW_FILE_STORE`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

DEFAULT_EXPERIMENT = "fraud_detector"
TRACKING_DIR = "mlruns"
TRACKING_DB = "mlflow.db"


def default_tracking_uri(root: Path) -> str:
    """URI SQLite dentro de `mlruns/`, anclado al root del proyecto."""
    return "sqlite:///" + (root / TRACKING_DIR / TRACKING_DB).as_posix()


def configure_tracking(
    tracking_uri: str, experiment: str = DEFAULT_EXPERIMENT
) -> tuple[MlflowClient, str]:
    """Devuelve un cliente y el id del experimento, creandolo si no existe."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment)
    experiment_id = (
        existing.experiment_id
        if existing is not None
        else client.create_experiment(experiment)
    )
    return client, experiment_id


def _scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return None


@dataclass
class Tracker:
    """Registro de runs idempotente por `tags.tracking_key`.

    Un backfill se puede re-ejecutar sin duplicar runs: si ya existe un run con
    la misma clave se salta, salvo `force=True`.
    """

    client: MlflowClient
    experiment_id: str
    root: Path
    force: bool = False
    include_models: bool = True
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def ensure_run(
        self,
        key: str,
        name: str,
        tags: dict[str, str] | None = None,
        parent_run_id: str | None = None,
    ) -> tuple[str, bool]:
        """Crea el run o devuelve el existente. `(run_id, creado)`."""
        found = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.tracking_key = '{key}'",
            max_results=1,
        )
        if found and not self.force:
            return found[0].info.run_id, False
        merged = {**(tags or {}), "tracking_key": key}
        if parent_run_id is not None:
            merged["mlflow.parentRunId"] = parent_run_id
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=name,
            tags=merged,
        )
        return run.info.run_id, True

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        from mlflow.entities import Param

        clean = {k: _scalar(v) for k, v in params.items() if _scalar(v) is not None}
        batch = [Param(k, str(v)) for k, v in clean.items()]
        if batch:
            self.client.log_batch(run_id, params=batch)

    def log_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        from mlflow.entities import Metric

        batch = [
            Metric(k, float(v), 0, 0)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if batch:
            self.client.log_batch(run_id, metrics=batch)

    def log_path(
        self, run_id: str, path: Path, artifact_path: str | None = None
    ) -> bool:
        if not path.exists():
            self.missing.append(path.as_posix())
            return False
        if path.is_dir():
            self.client.log_artifacts(run_id, str(path), artifact_path=artifact_path)
        else:
            self.client.log_artifact(run_id, str(path), artifact_path=artifact_path)
        return True

    def log_model(self, run_id: str, model_path: Path) -> bool:
        """Registra el estimador en formato MLflow (flavor LightGBM)."""
        if not self.include_models:
            return False
        if not model_path.exists():
            self.missing.append(model_path.as_posix())
            return False
        import tempfile

        import joblib
        import mlflow.lightgbm

        estimator = joblib.load(model_path)
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.lightgbm.save_model(estimator, path=tmp)
            self.client.log_artifacts(run_id, tmp, artifact_path="model")
        return True

    def finish(self, run_id: str, status: str = "FINISHED") -> None:
        self.client.set_terminated(run_id, status)

    def record(self, run_id: str, created: bool) -> None:
        (self.created if created else self.skipped).append(run_id)
