from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

ID_COLS = ["TransactionID"]
TARGET_COL = "isFraud"
PASSTHROUGH_COLS = ["TransactionDT"]
AMT_COL = "TransactionAmt"
DROP_COLS = ["dist2", "D7"]
CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "P_emaildomain",
    "R_emaildomain",
    "card1",
    "card2",
    "card3",
    "addr1",
    "addr2",
]


@dataclass
class Roles:
    numeric: list[str]
    amt: list[str]
    categorical: list[str]
    passthrough: list[str]
    drop: list[str] = field(default_factory=list)


def get_roles(df: pd.DataFrame) -> Roles:
    """Asigna cada columna de features a su rol y valida las requeridas."""
    required = [
        c
        for c in [*ID_COLS, TARGET_COL, *PASSTHROUGH_COLS, AMT_COL]
        if c not in df.columns
    ]
    if required:
        raise ValueError(f"Columnas requeridas ausentes: {required}")

    drop = [c for c in DROP_COLS if c in df.columns]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    excluded = (
        set(drop)
        | set(categorical)
        | {*ID_COLS, TARGET_COL, *PASSTHROUGH_COLS, AMT_COL}
    )
    numeric = [c for c in df.columns if c not in excluded]
    return Roles(
        numeric=numeric,
        amt=[AMT_COL],
        categorical=categorical,
        passthrough=passthrough,
        drop=drop,
    )


def build_pipeline(roles: Roles) -> ColumnTransformer:
    transformers = []
    if roles.numeric:
        num_imputer = SimpleImputer(strategy="median", add_indicator=True)
        transformers.append(("num", num_imputer, roles.numeric))
    if roles.amt:
        amt = Pipeline(
            steps=[
                (
                    "log1p",
                    FunctionTransformer(
                        np.log1p, validate=True, feature_names_out="one-to-one"
                    ),
                ),
                ("scale", RobustScaler()),
            ]
        )
        transformers.append(("amt", amt, roles.amt))
    if roles.categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        transformers.append(("cat", encoder, roles.categorical))
    if roles.passthrough:
        transformers.append(("raw", "passthrough", roles.passthrough))
    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


def _select_features(df: pd.DataFrame, roles: Roles) -> pd.DataFrame:
    keep = roles.numeric + roles.amt + roles.categorical + roles.passthrough
    return df[list(keep)].copy()


def fit_transform(
    df: pd.DataFrame, pipeline: ColumnTransformer, roles: Roles
) -> pd.DataFrame:
    X = _select_features(df, roles)
    values = pipeline.fit_transform(X)
    columns = list(pipeline.get_feature_names_out())
    return pd.DataFrame(values, columns=columns, index=X.index)


def transform(
    df: pd.DataFrame, pipeline: ColumnTransformer, roles: Roles
) -> pd.DataFrame:
    X = _select_features(df, roles)
    values = pipeline.transform(X)
    columns = list(pipeline.get_feature_names_out())
    return pd.DataFrame(values, columns=columns, index=X.index)


def preprocess_dataset(
    df: pd.DataFrame, pipeline: ColumnTransformer | None = None
) -> tuple[Roles, ColumnTransformer, pd.DataFrame]:
    """Aplica el pipeline y devuelve (roles, pipeline, frame con ID+target+features)."""
    roles = get_roles(df)
    if pipeline is None:
        pipeline = build_pipeline(roles)
        features = fit_transform(df, pipeline, roles)
    else:
        features = transform(df, pipeline, roles)

    meta = df[[*ID_COLS, TARGET_COL]].copy()
    out = pd.concat([meta, features.astype("float32")], axis=1)
    return roles, pipeline, out
