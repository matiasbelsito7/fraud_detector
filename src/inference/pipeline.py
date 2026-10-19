"""Pipeline de inferencia consistente con el entrenamiento (Fase L).

Una transaccion nueva atraviesa exactamente el mismo procesamiento que en
entrenamiento:

1. `preprocessor.joblib` (imputacion, encoding, scaling, passthrough).
2. `compute_features` (tiempo, frecuencia, historial, entidades, cliente).
3. el estimador serializado, con el umbral operativo configurado.

El orden final de columnas se contrasta contra `feature_name_` del modelo: si
falta o sobra una columna, la inferencia aborta en vez de predecir con una
matriz distinta a la del entrenamiento (`constitution.md` §8, `specs.md` §11).

**Historial.** Las features historicas necesitan transacciones previas. Si no se
pasa `history`, se generan con centinelas exactamente como en entrenamiento
(`cnt_* = 0`, medias previas `0.0`, `gap_sec_card1 = -1`); el resultado es valido
pero pierde la senal de historial. En produccion debe alimentarse el historial
del cliente disponible hasta la transaccion, nunca filas futuras: el pipeline
rechaza `TransactionDT` posterior al de la transaccion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.columns import AMT, CARD1, DT, KEY
from src.features.features import compute_features

TARGET_MARKER = "__inference_target__"

#: Valor centinela para categoricas ausentes. `OrdinalEncoder` lo trata como
#: desconocido (`unknown_value=-1`), el mismo resultado que `encoded_missing_value`
#: en entrenamiento. Se evita pasar NaN float a un encoder de categorias string.
CATEGORICAL_MISSING = "__inference_missing__"


def _branch_columns(preprocessor: Any, name: str) -> list[str]:
    for branch, _transformer, columns in getattr(preprocessor, "transformers_", []):
        if branch == name:
            return list(columns)
    return []


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Deployment:
    """Configuracion desplegable: artefacto, umbral y procedencia."""

    root: Path
    variant: str
    model_path: Path
    preprocessor_path: Path
    model_sha256: str
    preprocessor_sha256: str
    threshold: float
    config_id: int
    rounds: int
    split_sha256: str
    threshold_source: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        root: str | Path | None = None,
        verify_digest: bool = True,
    ) -> Deployment:
        root_path = (
            Path(root) if root is not None else Path(path).resolve().parent.parent
        )
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = root_path / config_path
        data = json.loads(config_path.read_text(encoding="utf-8"))

        model_path = root_path / data["model_artifact"]
        preprocessor_path = root_path / data["preprocessor_artifact"]
        if not model_path.exists():
            raise FileNotFoundError(f"Falta el modelo desplegable: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Falta el preprocessor desplegable: {preprocessor_path}"
            )

        threshold = float(data["threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Umbral fuera de (0, 1): {threshold}")

        if verify_digest:
            actual_model = file_digest(model_path)
            if actual_model != data["model_sha256"]:
                raise ValueError(
                    "El modelo no coincide con el sha256 de deployment.json "
                    f"({actual_model[:16]} != {data['model_sha256'][:16]}). "
                    "Regenera el artefacto o actualiza la configuracion."
                )
            actual_pre = file_digest(preprocessor_path)
            if actual_pre != data["preprocessor_sha256"]:
                raise ValueError(
                    "El preprocessor no coincide con el sha256 de deployment.json "
                    f"({actual_pre[:16]} != {data['preprocessor_sha256'][:16]})"
                )

        return cls(
            root=root_path,
            variant=str(data["variant"]),
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            model_sha256=str(data["model_sha256"]),
            preprocessor_sha256=str(data["preprocessor_sha256"]),
            threshold=threshold,
            config_id=int(data["config_id"]),
            rounds=int(data["rounds"]),
            split_sha256=str(data["split_sha256"]),
            threshold_source=str(data.get("threshold_source", "")),
        )


@dataclass(frozen=True)
class Prediction:
    """Resultado interpretable de una transaccion."""

    probability: float
    is_fraud: bool
    threshold: float
    variant: str
    config_id: int
    n_history: int

    @property
    def label(self) -> str:
        return "fraude" if self.is_fraud else "legitima"

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "is_fraud": self.is_fraud,
            "label": self.label,
            "threshold": self.threshold,
            "variant": self.variant,
            "config_id": self.config_id,
            "n_history": self.n_history,
        }


class InferencePipeline:
    """Preprocessor + feature engineering + modelo + umbral, sin estado global."""

    def __init__(
        self,
        deployment: Deployment,
        preprocessor: Any,
        model: Any,
        input_columns: list[str],
        feature_order: list[str],
    ) -> None:
        self.deployment = deployment
        self.preprocessor = preprocessor
        self.model = model
        self.input_columns = input_columns
        self.feature_order = feature_order
        self.raw_columns = list(dict.fromkeys([*input_columns, KEY]))
        self.categorical_columns = _branch_columns(preprocessor, "cat")

    @classmethod
    def load(
        cls,
        deployment_path: str | Path = "config/deployment.json",
        root: str | Path | None = None,
        verify_digest: bool = True,
    ) -> InferencePipeline:
        deployment = Deployment.load(
            deployment_path, root=root, verify_digest=verify_digest
        )
        preprocessor = joblib.load(deployment.preprocessor_path)
        model = joblib.load(deployment.model_path)
        input_columns = list(getattr(preprocessor, "feature_names_in_", []))
        if not input_columns:
            raise ValueError(
                "El preprocessor no registra feature_names_in_; no se puede "
                "reconstruir el esquema de entrada"
            )
        feature_order = list(
            getattr(model, "feature_name_", None)
            or getattr(model, "feature_names_in_", [])
        )
        if not feature_order:
            raise ValueError("El modelo no expone el nombre de sus features")
        return cls(
            deployment=deployment,
            preprocessor=preprocessor,
            model=model,
            input_columns=input_columns,
            feature_order=feature_order,
        )

    def predict(
        self,
        transaction: Mapping[str, Any] | pd.Series | pd.DataFrame,
        history: pd.DataFrame | None = None,
        identity: pd.DataFrame | None = None,
        history_identity: pd.DataFrame | None = None,
        threshold: float | None = None,
    ) -> Prediction:
        """Predice una transaccion; `history` debe ser estrictamente pasado."""
        cut = self.deployment.threshold if threshold is None else float(threshold)
        if not 0.0 < cut < 1.0:
            raise ValueError(f"Umbral fuera de (0, 1): {cut}")

        txn = self._transaction_frame(transaction)
        txn_dt = int(txn[DT].iloc[0])
        hist = self._history_frame(history, txn_dt)

        combined = (
            txn.copy() if len(hist) == 0 else pd.concat([hist, txn], ignore_index=True)
        )
        combined = combined.reset_index(drop=True).copy()
        combined[TARGET_MARKER] = False
        combined.loc[combined.index[-1], TARGET_MARKER] = True
        ordered = combined.sort_values(DT, kind="stable").reset_index(drop=True)
        target_rows = ordered.index[ordered[TARGET_MARKER]].tolist()
        if len(target_rows) != 1:
            raise ValueError("No se pudo identificar la transaccion a puntuar")
        target_idx = int(target_rows[0])

        identity_all = self._identity_frame(identity, history_identity)
        if identity_all is not None and (
            KEY not in txn.columns or pd.isna(txn[KEY].iloc[0])
        ):
            raise ValueError(
                "identity requiere que la transaccion traiga TransactionID"
            )

        pre_input = ordered.loc[[target_idx], self.input_columns].copy()
        for column in self.categorical_columns:
            if column in pre_input.columns:
                pre_input[column] = (
                    pre_input[column].astype("object").fillna(CATEGORICAL_MISSING)
                )
        values = self.preprocessor.transform(pre_input)
        processed = pd.DataFrame(
            values,
            columns=list(self.preprocessor.get_feature_names_out()),
            index=[target_idx],
        )

        features = compute_features(ordered, identity_all).loc[[target_idx]]
        X = pd.concat([processed, features], axis=1)

        missing = [c for c in self.feature_order if c not in X.columns]
        extra = [c for c in X.columns if c not in self.feature_order]
        if missing or extra:
            raise ValueError(
                "La matriz de inferencia no coincide con la del entrenamiento. "
                f"Faltan={missing[:5]} ({len(missing)}), "
                f"sobran={extra[:5]} ({len(extra)})"
            )
        X = X[self.feature_order]
        # El parquet de entrenamiento guarda las features como float32; se
        # replica ese dtype para que el binning de LightGBM sea el mismo.
        X = X.astype("float32")

        probability = float(np.asarray(self.model.predict_proba(X))[0, 1])
        return Prediction(
            probability=probability,
            is_fraud=bool(probability >= cut),
            threshold=cut,
            variant=self.deployment.variant,
            config_id=self.deployment.config_id,
            n_history=int(len(hist)),
        )

    def _transaction_frame(
        self, transaction: Mapping[str, Any] | pd.Series | pd.DataFrame
    ) -> pd.DataFrame:
        if isinstance(transaction, pd.DataFrame):
            if len(transaction) != 1:
                raise ValueError(
                    "predict() recibe UNA transaccion; para lotes, itera o usa "
                    "el pipeline por lotes"
                )
            frame = transaction.copy()
        elif isinstance(transaction, pd.Series):
            # `to_frame().T` de una Series con dtypes mixtos deja todo en object;
            # construir desde el dict conserva el dtype de cada columna.
            frame = pd.DataFrame([transaction.to_dict()])
        else:
            frame = pd.DataFrame([dict(transaction)])

        required = [DT, AMT, CARD1]
        absent = [c for c in required if c not in frame.columns]
        if absent:
            raise ValueError(f"Transaccion sin columnas requeridas: {absent}")
        for column in required:
            if pd.isna(frame[column].iloc[0]):
                raise ValueError(f"Transaccion con {column} nulo")

        if KEY not in frame.columns:
            frame[KEY] = -1
        elif pd.isna(frame[KEY].iloc[0]):
            frame[KEY] = -1

        return frame.reindex(columns=self.raw_columns)

    def _history_frame(
        self, history: pd.DataFrame | None, transaction_dt: int
    ) -> pd.DataFrame:
        if history is None or len(history) == 0:
            return pd.DataFrame(columns=self.raw_columns)
        if DT not in history.columns:
            raise ValueError(f"El historial no trae {DT}")
        future = history[DT] > transaction_dt
        if bool(future.any()):
            raise ValueError(
                "El historial contiene transacciones posteriores a la que se "
                "quiere puntuar (leakage temporal, constitution.md §3): "
                f"{int(future.sum())} filas con {DT} > {transaction_dt}"
            )
        frame = history.reindex(columns=self.raw_columns)
        if KEY not in history.columns:
            frame[KEY] = np.nan
        return frame

    @staticmethod
    def _identity_frame(
        identity: pd.DataFrame | None, history_identity: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        frames = [f for f in (history_identity, identity) if f is not None]
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        if KEY not in combined.columns:
            raise ValueError(f"identity no trae {KEY}")
        return combined.drop_duplicates(subset=[KEY], keep="last")
