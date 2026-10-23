"""Monitorizacion de drift sobre las features que consume el modelo.

La referencia se construye una sola vez a partir de `train` y describe, por
feature, los bins (cuantiles) y sus proporciones. En produccion, cada prediccion
aporta una observacion a un monitor en memoria que calcula PSI contra esa
referencia. No hay ground truth: esto mide drift de entrada, no calidad del
modelo.

El monitor recibe exactamente la matriz preprocesada que ve el modelo, no la
transaccion raw. Asi el drift medido es el que puede afectar a la prediccion y no
depende de como se serialice una categoria en el body HTTP.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
DEFAULT_BINS = 10
DEFAULT_WINDOW = 10_000
EPSILON = 1e-6
REFERENCE_VERSION = 1


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], *, epsilon: float = EPSILON
) -> float:
    """PSI entre dos distribuciones de la misma longitud.

    Los buckets vacios en ambas distribuciones no aportan. Si un bucket solo
    existe en una, se usa `epsilon` para evitar log(0): ese caso es precisamente
    el que debe disparar la metrica.
    """
    expected_a = np.asarray(expected, dtype=float)
    actual_a = np.asarray(actual, dtype=float)
    if expected_a.shape != actual_a.shape:
        raise ValueError(
            f"Distribuciones de distinta longitud: {expected_a.shape} vs "
            f"{actual_a.shape}"
        )
    if expected_a.size == 0:
        raise ValueError("Sin buckets no se puede calcular PSI")
    keep = (expected_a > 0.0) | (actual_a > 0.0)
    if not keep.any():
        raise ValueError("Ambas distribuciones estan vacias")
    expected_a = np.clip(expected_a[keep], epsilon, None)
    actual_a = np.clip(actual_a[keep], epsilon, None)
    return float(np.sum((actual_a - expected_a) * np.log(actual_a / expected_a)))


def _numeric(values: Any) -> np.ndarray:
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _edges_and_proportions(
    values: np.ndarray, bins: int
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    finite = values[np.isfinite(values)]
    missing_rate = float(1.0 - finite.size / values.size) if values.size else 0.0
    if finite.size == 0:
        return (), (0.0, 1.0), 1.0

    unique = np.unique(finite)
    if unique.size == 1:
        value = float(unique[0])
        lower = float(np.nextafter(value, -np.inf))
        upper = float(np.nextafter(value, np.inf))
        if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
            edges = np.array([lower, upper], dtype=float)
        else:
            edges = np.array([], dtype=float)
    elif unique.size <= bins:
        edges = unique[1:]
    else:
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1])
        edges = np.unique(quantiles)

    if edges.size:
        counts = np.histogram(
            finite, bins=np.concatenate([[-np.inf], edges, [np.inf]])
        )[0]
    else:
        counts = np.array([finite.size], dtype=int)
    finite_share = 1.0 - missing_rate
    proportions = counts.astype(float) / finite.size * finite_share
    with_missing = np.concatenate([proportions, [missing_rate]])
    return (
        tuple(float(edge) for edge in edges),
        tuple(float(value) for value in with_missing),
        missing_rate,
    )


@dataclass(frozen=True)
class FeatureProfile:
    """Bins y proporciones de referencia de una feature numerica."""

    name: str
    edges: tuple[float, ...]
    proportions: tuple[float, ...]
    missing_rate: float
    n: int

    def __post_init__(self) -> None:
        if len(self.proportions) != len(self.edges) + 2:
            raise ValueError(
                f"{self.name}: proporciones={len(self.proportions)} no coincide "
                f"con edges={len(self.edges)} + 2"
            )
        if not math.isclose(sum(self.proportions), 1.0, abs_tol=1e-6):
            raise ValueError(f"{self.name}: las proporciones no suman 1")
        if not 0.0 <= self.missing_rate <= 1.0:
            raise ValueError(f"{self.name}: missing_rate fuera de [0, 1]")

    @property
    def n_bins(self) -> int:
        return len(self.proportions)

    def bucket(self, value: Any) -> int:
        """Indice de bin; el ultimo bucket es missing/no numerico."""
        if value is None:
            return self.n_bins - 1
        try:
            numeric = float(value)
        except TypeError, ValueError:
            return self.n_bins - 1
        if not math.isfinite(numeric):
            return self.n_bins - 1
        return int(np.searchsorted(self.edges, numeric, side="right"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "edges": list(self.edges),
            "proportions": list(self.proportions),
            "missing_rate": self.missing_rate,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureProfile:
        return cls(
            name=str(data["name"]),
            edges=tuple(float(value) for value in data["edges"]),
            proportions=tuple(float(value) for value in data["proportions"]),
            missing_rate=float(data["missing_rate"]),
            n=int(data["n"]),
        )


def build_profile(
    name: str, values: Any, *, bins: int = DEFAULT_BINS
) -> FeatureProfile:
    if bins < 2:
        raise ValueError("bins debe ser >= 2")
    array = _numeric(values)
    edges, proportions, missing_rate = _edges_and_proportions(array, bins)
    return FeatureProfile(
        name=name,
        edges=edges,
        proportions=proportions,
        missing_rate=missing_rate,
        n=int(array.size),
    )


@dataclass(frozen=True)
class DriftReference:
    """Referencia completa de drift, serializable a JSON."""

    features: tuple[FeatureProfile, ...]
    metadata: Mapping[str, Any]
    bins: int = DEFAULT_BINS

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        features: Sequence[str],
        *,
        bins: int = DEFAULT_BINS,
        metadata: Mapping[str, Any] | None = None,
    ) -> DriftReference:
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"Features de referencia ausentes: {missing[:10]}")
        profiles = tuple(
            build_profile(name, frame[name], bins=bins) for name in features
        )
        return cls(
            features=profiles,
            metadata=dict(metadata or {}),
            bins=bins,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REFERENCE_VERSION,
            "bins": self.bins,
            "metadata": dict(self.metadata),
            "features": [profile.to_dict() for profile in self.features],
        }
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> DriftReference:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(data.get("version", REFERENCE_VERSION))
        if version != REFERENCE_VERSION:
            raise ValueError(f"Version de referencia de drift no soportada: {version}")
        features = tuple(FeatureProfile.from_dict(item) for item in data["features"])
        if not features:
            raise ValueError("La referencia de drift no contiene features")
        return cls(
            features=features,
            metadata=dict(data.get("metadata", {})),
            bins=int(data.get("bins", DEFAULT_BINS)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": REFERENCE_VERSION,
            "bins": self.bins,
            "metadata": dict(self.metadata),
            "features": [profile.to_dict() for profile in self.features],
        }


def _status(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def build_reference(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    bins: int = DEFAULT_BINS,
    metadata: Mapping[str, Any] | None = None,
) -> DriftReference:
    """Atajo de `DriftReference.build` para scripts y tests."""
    return DriftReference.build(frame, features, bins=bins, metadata=metadata)


class DriftMonitor:
    """Acumula observaciones en memoria y calcula PSI contra la referencia.

    La ventana es exponencial: al alcanzar `max_observations`, se conserva la
    mitad de los conteos. Asi el monitor no crece sin limite y el PSI refleja el
    trafico reciente. El estado es por proceso y no se persiste.
    """

    def __init__(
        self,
        reference: DriftReference,
        *,
        max_observations: int = DEFAULT_WINDOW,
        moderate: float = PSI_MODERATE,
        significant: float = PSI_SIGNIFICANT,
    ) -> None:
        if max_observations <= 0:
            raise ValueError("max_observations debe ser positivo")
        if not 0.0 < moderate < significant:
            raise ValueError("Se exige 0 < moderate < significant")
        self.reference = reference
        self.max_observations = int(max_observations)
        self.moderate = float(moderate)
        self.significant = float(significant)
        self._counts = {
            profile.name: np.zeros(profile.n_bins, dtype=float)
            for profile in reference.features
        }
        self._total = 0.0
        self._lock = threading.Lock()

    @property
    def n_observations(self) -> int:
        return int(round(self._total))

    def observe(self, row: Mapping[str, Any] | pd.Series) -> None:
        with self._lock:
            for profile in self.reference.features:
                value = row.get(profile.name)
                self._counts[profile.name][profile.bucket(value)] += 1.0
            self._total += 1.0
            self._decay()

    def observe_frame(self, frame: pd.DataFrame, *, limit: int | None = None) -> None:
        if frame.empty:
            return
        if limit is not None and len(frame) > limit:
            frame = frame.iloc[:limit]
        with self._lock:
            for profile in self.reference.features:
                if profile.name in frame.columns:
                    values = _numeric(frame[profile.name])
                else:
                    values = np.full(len(frame), np.nan)
                indices = np.searchsorted(profile.edges, values, side="right")
                indices = np.where(np.isfinite(values), indices, profile.n_bins - 1)
                self._counts[profile.name] += np.bincount(
                    indices, minlength=profile.n_bins
                )
            self._total += float(len(frame))
            self._decay()

    def _decay(self) -> None:
        while self._total > self.max_observations and self._total > 0:
            for counts in self._counts.values():
                counts *= 0.5
            self._total *= 0.5

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for profile in self.reference.features:
                counts = self._counts[profile.name]
                total = float(counts.sum())
                actual = counts / total if total > 0 else np.zeros_like(counts, float)
                if total > 0:
                    psi = population_stability_index(
                        profile.proportions, [float(value) for value in actual]
                    )
                    status = _status(psi)
                else:
                    psi = float("nan")
                    status = "no_data"
                rows.append(
                    {
                        "feature": profile.name,
                        "psi": None if math.isnan(psi) else round(psi, 6),
                        "status": status,
                        "reference_missing_rate": profile.missing_rate,
                        "observed_missing_rate": (
                            float(actual[-1]) if total > 0 else None
                        ),
                        "n_observed": int(round(total)),
                    }
                )
            rows.sort(
                key=lambda item: (
                    item["psi"] is None,
                    -(item["psi"] or 0.0),
                    item["feature"],
                )
            )
            values = [row["psi"] for row in rows if row["psi"] is not None]
            if not values:
                overall = "no_data"
            elif any(value >= self.significant for value in values):
                overall = "significant"
            elif any(value >= self.moderate for value in values):
                overall = "moderate"
            else:
                overall = "stable"
            return {
                "n_observations": self.n_observations,
                "max_observations": self.max_observations,
                "max_psi": round(max(values), 6) if values else None,
                "status": overall,
                "thresholds": {
                    "moderate": self.moderate,
                    "significant": self.significant,
                },
                "reference": {
                    "source": self.reference.metadata.get("source"),
                    "split_sha256": self.reference.metadata.get("split_sha256"),
                    "n_rows": self.reference.metadata.get("n_rows_sampled")
                    or self.reference.metadata.get("n_rows"),
                },
                "features": rows,
            }

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        max_observations: int = DEFAULT_WINDOW,
    ) -> DriftMonitor:
        return cls(
            DriftReference.load(path),
            max_observations=max_observations,
        )
