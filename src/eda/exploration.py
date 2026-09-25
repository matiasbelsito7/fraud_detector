from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.validation import TARGET, load_identity, load_transaction

DAY_SEC = 86400
WEEK_SEC = 7 * DAY_SEC

NUMERIC_FAMILY = {
    "C1-C14": [f"C{i}" for i in range(1, 15)],
    "D1-D15": [f"D{i}" for i in range(1, 16)],
    "V1-V339": [f"V{i}" for i in range(1, 340)],
    "id01-id38": [f"id_{i:02d}" for i in range(1, 39)],
}

CATEGORY_FAMILY = {
    "producto": ["ProductCD"],
    "tarjeta": [f"card{i}" for i in range(1, 7)],
    "direccion": ["addr1", "addr2"],
    "email": ["P_emaildomain", "R_emaildomain"],
    "m1_m9": [f"M{i}" for i in range(1, 10)],
    "dispositivo": ["DeviceType", "DeviceInfo"],
}


def _dtype_groups(df: pd.DataFrame) -> dict:
    dtypes = df.dtypes.astype(str)
    return {
        "numerico": int(dtypes.isin(["float64", "int64"]).sum()),
        "string": int(dtypes.isin(["object", "str"]).sum()),
        "bool": int((dtypes == "bool").sum()),
    }


def _missing_analysis(df: pd.DataFrame) -> dict:
    n = len(df)
    miss = df.isna().sum()
    row = (
        pd.DataFrame(
            {
                "columna": miss.index,
                "missing": miss.values,
                "pct": (miss.values / n * 100).round(2),
            }
        )
        .sort_values("pct", ascending=False)
        .reset_index(drop=True)
    )
    buckets = {
        "sin_missing": int((row["missing"] == 0).sum()),
        "0_10": int(((row["pct"] > 0) & (row["pct"] <= 10)).sum()),
        "10_50": int(((row["pct"] > 10) & (row["pct"] <= 50)).sum()),
        "50_90": int(((row["pct"] > 50) & (row["pct"] <= 90)).sum()),
        "90_100": int((row["pct"] > 90).sum()),
    }
    family_missing = {}
    for family, cols in NUMERIC_FAMILY.items():
        present = [c for c in cols if c in df.columns]
        if present:
            family_missing[family] = round(
                float(df[present].isna().mean().mean() * 100), 2
            )
    return {
        "columnas": len(row),
        "buckets": buckets,
        "top20": row.head(20).to_dict("records"),
        "family_mean_pct": family_missing,
        "lista_columns_con_10pct_missing": row.loc[row["pct"] > 10, "columna"].tolist(),
    }


def _cardinality(df: pd.DataFrame) -> list[dict]:
    obj_cols = df.select_dtypes(include=["object"]).columns
    out = []
    for c in obj_cols:
        out.append(
            {
                "columna": c,
                "nunique": int(df[c].nunique(dropna=True)),
                "missing_pct": round(100 * df[c].isna().mean(), 2),
            }
        )
    return sorted(out, key=lambda r: r["nunique"], reverse=True)


def _corr_with_target(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    numeric_cols = [
        c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr = df[numeric_cols].corrwith(df[TARGET]).dropna()
    top = corr.abs().sort_values(ascending=False).head(top_n)
    return [{"columna": c, "corr": round(float(corr[c]), 4)} for c in top.index]


def _amount_analysis(txn: pd.DataFrame) -> dict:
    amt = txn["TransactionAmt"].dropna()
    by_class = (
        txn[["TransactionAmt", TARGET]]
        .groupby(TARGET)["TransactionAmt"]
        .describe()
        .round(2)
    )
    return {
        "overall": amt.describe().round(2).to_dict(),
        "by_class": by_class.to_dict("index"),
        "no_positivos": int((amt <= 0).sum()),
        "missing": int(txn["TransactionAmt"].isna().sum()),
    }


def _time_analysis(txn: pd.DataFrame) -> dict:
    dt = txn["TransactionDT"].astype("int64")
    week = dt // WEEK_SEC
    hour = (dt % DAY_SEC) // 3600
    per_week = txn.assign(week=week).groupby("week")[TARGET].agg(["size", "sum"])
    per_week["fraud_rate"] = (100 * per_week["sum"] / per_week["size"]).round(3)
    per_hour = pd.DataFrame({"week": week, "hour": hour, "y": txn[TARGET]})
    hourly = per_hour.groupby("hour")["y"].agg(["size", "sum"])
    hourly["fraud_rate"] = (100 * hourly["sum"] / hourly["size"]).round(3)
    return {
        "min_dt": int(dt.min()),
        "max_dt": int(dt.max()),
        "dias_span": round((dt.max() - dt.min()) / DAY_SEC, 1),
        "fraud_rate_overall_pct": round(100 * txn[TARGET].mean(), 3),
        "fraud_rate_min_week_pct": round(float(per_week["fraud_rate"].min()), 3),
        "fraud_rate_max_week_pct": round(float(per_week["fraud_rate"].max()), 3),
        "fraud_rate_p90_week_pct": round(
            float(per_week["fraud_rate"].quantile(0.9)), 3
        ),
        "hora_min_fraud_pct": round(float(hourly["fraud_rate"].min()), 3),
        "hora_max_fraud_pct": round(float(hourly["fraud_rate"].max()), 3),
        "hora_menor_rate": int(hourly["fraud_rate"].idxmin()),
        "hora_mayor_rate": int(hourly["fraud_rate"].idxmax()),
    }


def _category_fraud_rates(df: pd.DataFrame, top_n: int = 8) -> list[dict]:
    out = []
    for name, cols in CATEGORY_FAMILY.items():
        present = [c for c in cols if c in df.columns]
        for c in present:
            grouped = (
                df[[c, TARGET]]
                .dropna(subset=[c])
                .groupby(c)[TARGET]
                .agg(["size", "sum", "mean"])
            )
            grouped = (
                grouped[grouped["size"] >= 100]
                .sort_values("size", ascending=False)
                .head(top_n)
            )
            out.append(
                {
                    "familia": name,
                    "columna": c,
                    "top": [
                        {
                            "valor": str(idx),
                            "n": int(r["size"]),
                            "fraud_pct": round(100 * r["mean"], 2),
                        }
                        for idx, r in grouped.iterrows()
                    ],
                }
            )
    return out


def _constant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def _plot_missing(top: list[dict], path: Path) -> None:
    df = pd.DataFrame(top)
    if df.empty:
        return
    df = df.head(30)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["columna"], df["pct"], color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("% de valores faltantes")
    ax.set_title("Top 30 columnas con mayor % de missing (transaction)")
    ax.axvline(50, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_target(txn: pd.DataFrame, path: Path) -> None:
    counts = txn[TARGET].value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=["#1f77b4", "#d62728"])
    ax.set_ylabel("% del total")
    ax.set_title(f"Distribucion de {TARGET} (fraude = 1)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, f"{v:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_amount(txn: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in (0, 1):
        data = txn.loc[txn[TARGET] == label, "TransactionAmt"].dropna()
        data = data[(data > 0) & (data < 1000)]
        ax.hist(
            data,
            bins=50,
            alpha=0.5,
            label=f"label={label} (n={len(data)})",
            log=True,
        )
    ax.set_xlabel("TransactionAmt (USD)")
    ax.set_ylabel("frecuencia (log)")
    ax.set_title("Distribucion de TransactionAmt por clase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_time(txn: pd.DataFrame, path: Path) -> None:
    dt = txn["TransactionDT"].astype("int64")
    week = dt // WEEK_SEC
    per_week = txn.assign(week=week).groupby("week")[TARGET].agg(["size", "sum"])
    per_week["fraud_rate"] = 100 * per_week["sum"] / per_week["size"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(per_week.index, per_week["size"], color="#1f77b4")
    ax[0].set_ylabel("transacciones")
    ax[0].set_title("Volumen de transacciones por semana")
    ax[1].plot(per_week.index, per_week["fraud_rate"], color="#d62728")
    ax[1].axhline(100 * txn[TARGET].mean(), color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("indice de semana (desde inicio)")
    ax[1].set_ylabel("fraude %")
    ax[1].set_title("Tasa de fraude por semana")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_hour(txn: pd.DataFrame, path: Path) -> None:
    dt = txn["TransactionDT"].astype("int64")
    hour = (dt % DAY_SEC) // 3600
    hourly = txn.assign(hour=hour).groupby("hour")[TARGET].agg(["size", "sum"])
    hourly["fraud_rate"] = 100 * hourly["sum"] / hourly["size"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].bar(hourly.index, hourly["size"], color="#1f77b4")
    ax[0].set_ylabel("transacciones")
    ax[0].set_title("Volumen de transacciones por hora del dia")
    ax[1].bar(hourly.index, hourly["fraud_rate"], color="#d62728")
    ax[1].set_xlabel("hora (0-23)")
    ax[1].set_ylabel("fraude %")
    ax[1].set_title("Tasa de fraude por hora del dia")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_corr(top: list[dict], path: Path) -> None:
    cols = [r["columna"] for r in reversed(top)]
    vals = [r["corr"] for r in reversed(top)]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
    ax.barh(cols, vals, color=colors)
    ax.set_xlabel("correlacion con isFraud")
    ax.set_title("Top 20 correlaciones con el target")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_eda(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Cargando datos ...")
    txn = load_transaction(data_dir)
    identity = load_identity(data_dir)
    merged = txn.merge(identity, on="TransactionID", how="left")

    print("[2/7] Analisis de tipos y missing ...")
    dtype_groups = _dtype_groups(txn)
    missing = _missing_analysis(txn)
    missing_identity = _missing_analysis(identity)

    print("[3/7] Cardinalidad y columnas constantes ...")
    cardinality = _cardinality(txn)
    constant_cols = _constant_columns(txn)

    print("[4/7] Analisis de target y correlaciones ...")
    target_counts = txn[TARGET].value_counts().to_dict()
    target_pct = round(100 * txn[TARGET].mean(), 3)
    corr = _corr_with_target(txn)

    print("[5/7] Analisis de importes ...")
    amount = _amount_analysis(txn)

    print("[6/7] Analisis temporal ...")
    time_analysis = _time_analysis(txn)

    print("[7/7] Tasas de fraude por categoria ...")
    category_rates = _category_fraud_rates(merged)
    missing_sen = int(merged["DeviceType"].isna().sum())
    devtype = (
        merged[["DeviceType", TARGET]]
        .dropna(subset=["DeviceType"])
        .groupby("DeviceType")[TARGET]
        .agg(["size", "mean"])
    )
    devtype_fraud = [
        {"valor": str(idx), "n": int(r["size"]), "fraud_pct": round(100 * r["mean"], 2)}
        for idx, r in devtype.sort_values("size", ascending=False).iterrows()
    ]

    print("Generando graficos ...")
    _plot_target(txn, out_dir / "png_target.png")
    _plot_amount(txn, out_dir / "png_amount.png")
    _plot_time(txn, out_dir / "png_time.png")
    _plot_hour(txn, out_dir / "png_hour.png")
    _plot_corr(corr, out_dir / "png_corr.png")
    _plot_missing(missing["top20"], out_dir / "png_missing.png")
    product = next((r for r in category_rates if r["columna"] == "ProductCD"), None)
    if product:
        _plot_product(product["top"], out_dir / "png_product.png")

    result = {
        "dimensiones": {
            "transaction": {"filas": int(len(txn)), "columnas": int(txn.shape[1])},
            "identity": {
                "filas": int(len(identity)),
                "columnas": int(identity.shape[1]),
            },
            "dtype_groups_txn": dtype_groups,
        },
        "missing_transaction": missing,
        "missing_identity": missing_identity,
        "cardinality": cardinality,
        "constant_columns": constant_cols,
        "target": {
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "fraud_pct": target_pct,
        },
        "corr_top20": corr,
        "amount": amount,
        "time": time_analysis,
        "category_fraud_rates": category_rates,
        "device_type": {
            "cobertura_pct": round(100 * (1 - missing_sen / len(merged)), 2),
            "fraud_rate": devtype_fraud,
        },
        "graficas": [p.name for p in sorted(out_dir.glob("*.png"))],
    }
    return result


def _plot_product(top: list[dict], path: Path) -> None:
    vals = [r["valor"] for r in top]
    fraud = [r["fraud_pct"] for r in top]
    n = [r["n"] for r in top]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(vals, fraud, color="#d62728", alpha=0.8)
    ax1.set_ylabel("fraude %")
    ax1.set_title("Tasa de fraude por ProductCD (top por volumen)")
    ax2 = ax1.twinx()
    ax2.plot(vals, n, "o-", color="#1f77b4")
    ax2.set_ylabel("n transacciones (log)")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
