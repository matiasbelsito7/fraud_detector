from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    DriftMonitor,
    DriftReference,
    build_profile,
    build_reference,
    population_stability_index,
)


def test_psi_es_cero_para_la_misma_distribucion() -> None:
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_detecta_desplazamiento() -> None:
    psi = population_stability_index([0.5, 0.5], [0.1, 0.9])
    assert psi == pytest.approx(0.878889, abs=1e-5)
    assert psi >= 0.25


def test_psi_rechaza_longitudes_distintas() -> None:
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])


def test_perfil_maneja_missing() -> None:
    profile = build_profile("x", [1.0, 2.0, None, np.nan], bins=2)
    assert profile.missing_rate == pytest.approx(0.5)
    assert profile.proportions[-1] == pytest.approx(0.5)
    assert profile.bucket(None) == profile.n_bins - 1
    assert profile.bucket("no-numerico") == profile.n_bins - 1


def test_perfil_constante_detecta_cambio() -> None:
    profile = build_profile("x", [150.0, 150.0, 150.0])
    assert profile.bucket(150.0) == 1
    assert profile.bucket(150.0 + 1.0) == 2
    assert profile.bucket(150.0 - 1.0) == 0


def test_perfil_todo_missing() -> None:
    profile = build_profile("x", [None, None])
    assert profile.edges == ()
    assert profile.proportions == (0.0, 1.0)
    assert profile.bucket(None) == 1


def test_referencia_serializa_y_recarga(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 100),
            "b": [1.0, 2.0, 3.0, 4.0] * 25,
        }
    )
    reference = build_reference(
        frame,
        ["a", "b"],
        bins=4,
        metadata={"source": "train", "test_usado": False},
    )
    path = reference.save(tmp_path / "drift.json")
    loaded = DriftReference.load(path)
    assert loaded == reference
    assert loaded.metadata["source"] == "train"


def test_referencia_rechaza_feature_ausente() -> None:
    with pytest.raises(ValueError, match="ausentes"):
        build_reference(pd.DataFrame({"a": [1.0]}), ["a", "b"])


def test_monitor_estable_si_observa_la_referencia() -> None:
    frame = pd.DataFrame({"a": np.linspace(0.0, 1.0, 200)})
    reference = build_reference(frame, ["a"], bins=5)
    monitor = DriftMonitor(reference)
    monitor.observe_frame(frame)
    snapshot = monitor.snapshot()
    assert snapshot["n_observations"] == 200
    assert snapshot["status"] == "stable"
    assert snapshot["max_psi"] == pytest.approx(0.0, abs=1e-6)


def test_monitor_detecta_drift_significativo() -> None:
    reference = build_reference(pd.DataFrame({"a": np.zeros(100)}), ["a"])
    monitor = DriftMonitor(reference)
    monitor.observe_frame(pd.DataFrame({"a": np.full(100, 10.0)}))
    snapshot = monitor.snapshot()
    assert snapshot["status"] == "significant"
    assert snapshot["features"][0]["status"] == "significant"


def test_monitor_decae_ventana() -> None:
    reference = build_reference(pd.DataFrame({"a": np.arange(10.0)}), ["a"])
    monitor = DriftMonitor(reference, max_observations=2)
    for _ in range(3):
        monitor.observe({"a": 1.0})
    assert monitor.n_observations <= 2
    assert monitor.snapshot()["n_observations"] <= 2


def test_monitor_sin_observaciones() -> None:
    reference = build_reference(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])
    snapshot = DriftMonitor(reference).snapshot()
    assert snapshot["status"] == "no_data"
    assert snapshot["n_observations"] == 0
    assert snapshot["features"][0]["psi"] is None
