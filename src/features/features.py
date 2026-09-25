from __future__ import annotations

import numpy as np
import pandas as pd

KEY = "TransactionID"
DT = "TransactionDT"
AMT = "TransactionAmt"

_Group = str | list[str] | tuple[str, ...]


def _as_keys(group: _Group) -> list[str]:
    if isinstance(group, str):
        return [group]
    return list(group)


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hour = (df[DT] % 86400) // 3600
    dow = (df[DT] // 86400) % 7
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def prior_count(df: pd.DataFrame, group: _Group, name: str) -> pd.Series:
    return df.groupby(_as_keys(group), dropna=False).cumcount().rename(name)


def previous_expanding(
    df: pd.DataFrame, group: _Group, col: str, agg: str, name: str
) -> pd.Series:
    def _agg(s: pd.Series) -> pd.Series:
        return getattr(s.expanding(), agg)().shift(1)

    return df.groupby(_as_keys(group), dropna=False)[col].transform(_agg).rename(name)


def previous_value(df: pd.DataFrame, group: _Group, col: str, name: str) -> pd.Series:
    return df.groupby(_as_keys(group), dropna=False)[col].shift(1).rename(name)


def running_min_prev(df: pd.DataFrame, group: _Group, col: str) -> pd.Series:
    def _min(s: pd.Series) -> pd.Series:
        return s.cummin().shift(1)

    return df.groupby(_as_keys(group), dropna=False)[col].transform(_min)


def single_expanding_nunique_prev(
    df: pd.DataFrame, group: _Group, col: str
) -> pd.Series:
    def _nunique(s: pd.Series) -> pd.Series:
        return s.fillna(-1.0).expanding().nunique().shift(1)

    return df.groupby(_as_keys(group), dropna=False)[col].transform(_nunique)


def device_features(df: pd.DataFrame, identity: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    has = pd.Series(0, index=df.index, dtype=np.int64)
    mobile = pd.Series(-1, index=df.index, dtype=np.int64)
    if (
        identity is not None
        and KEY in identity.columns
        and "DeviceType" in identity.columns
    ):
        dev = df[[KEY]].merge(identity[[KEY, "DeviceType"]], on=KEY, how="left")[
            "DeviceType"
        ]
        has = dev.notna().astype(np.int64)
        mobile[dev == "mobile"] = 1
        mobile[dev == "desktop"] = 0
    out["has_identity"] = has
    out["is_mobile"] = mobile
    return out


def compute_features(
    df: pd.DataFrame, identity: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Genera features solo a partir de informacion pasada/simultanea (nunca futura)."""
    required = [KEY, DT, AMT, "card1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas ausentes: {missing}")

    ordered = df.sort_values(DT, kind="stable")

    mean_prev = previous_expanding(ordered, "card1", AMT, "mean", "mean_amt_card1_prev")
    std_prev = previous_expanding(ordered, "card1", AMT, "std", "std_amt_card1_prev")

    parts = [
        time_features(ordered),
        prior_count(ordered, "card1", "cnt_card1").to_frame(),
        prior_count(ordered, "addr1", "cnt_addr1").to_frame(),
        prior_count(ordered, "P_emaildomain", "cnt_P_emaildomain").to_frame(),
        prior_count(ordered, ("card1", "addr1"), "cnt_card1_addr1").to_frame(),
        mean_prev.to_frame(),
        std_prev.to_frame(),
    ]
    out = pd.concat(parts, axis=1)

    out["amt_rel_card1"] = (ordered[AMT] / out["mean_amt_card1_prev"]).fillna(1.0)
    out["mean_amt_card1_prev"] = out["mean_amt_card1_prev"].fillna(0.0)
    out["std_amt_card1_prev"] = out["std_amt_card1_prev"].fillna(0.0)

    gap_prev = previous_value(ordered, "card1", DT, "_gap")
    out["gap_sec_card1"] = (ordered[DT] - gap_prev).fillna(-1.0)

    min_prev = running_min_prev(ordered, "card1", DT)
    out["since_first_sec_card1"] = (ordered[DT] - min_prev).fillna(0.0)

    out["nunique_addr1_card1"] = single_expanding_nunique_prev(
        ordered, "card1", "addr1"
    ).fillna(0.0)

    p = ordered["P_emaildomain"]
    r = ordered["R_emaildomain"]
    both = p.notna() & r.notna()
    eq = pd.Series(-1, index=ordered.index, dtype=np.int64)
    eq[both] = (p[both] == r[both]).astype(np.int64)
    out["p_eq_r_email"] = eq

    out = pd.concat([out, device_features(ordered, identity)], axis=1)
    return out.loc[df.index]
