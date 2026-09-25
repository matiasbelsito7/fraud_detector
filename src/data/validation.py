from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

KEY = "TransactionID"
TARGET = "isFraud"


def load_transaction(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / TRANSACTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}")
    return pd.read_csv(path)


def load_identity(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / IDENTITY_FILE
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}")
    return pd.read_csv(path)


def _dtype_summary(df: pd.DataFrame) -> dict:
    counts = df.dtypes.astype(str).value_counts()
    return {k: int(v) for k, v in counts.items()}


def _structural_checks(txn: pd.DataFrame, identity: pd.DataFrame) -> list[dict]:
    checks = []
    key_txn = txn[KEY].dtype.name
    key_id = identity[KEY].dtype.name

    checks.append(
        {
            "check": "Tipo consistente de TransactionID",
            "pass": key_txn == key_id,
            "detail": f"transaction={key_txn}, identity={key_id}",
        }
    )
    checks.append(
        {
            "check": "TransactionID unico en transaction",
            "pass": txn[KEY].is_unique,
            "detail": f"filas={len(txn)}, valores_unicos={txn[KEY].nunique()}",
        }
    )
    checks.append(
        {
            "check": "TransactionID unico en identity",
            "pass": identity[KEY].is_unique,
            "detail": f"filas={len(identity)}, valores_unicos={identity[KEY].nunique()}",
        }
    )
    checks.append(
        {
            "check": "Sin valores nulos en TransactionID",
            "pass": not txn[KEY].isna().any() and not identity[KEY].isna().any(),
            "detail": f"txn_nulls={int(txn[KEY].isna().sum())}, identity_nulls={int(identity[KEY].isna().sum())}",
        }
    )
    checks.append(
        {
            "check": "TransactionID sin duplicados cruzados",
            "pass": identity[KEY].is_unique,
            "detail": f"identities={len(identity)}, count en identity de ids repetidos={int(identity[KEY].duplicated().sum())}",
        }
    )
    return checks


def _join_coverage(txn: pd.DataFrame, identity: pd.DataFrame) -> dict:
    tx_ids = set(txn[KEY])
    id_ids = set(identity[KEY])
    in_both = len(tx_ids.intersection(id_ids))
    only_identity = len(id_ids - tx_ids)
    return {
        "transacciones": int(len(txn)),
        "transacciones_con_identity": int(in_both),
        "coverage_pct": round(100.0 * in_both / len(txn), 2),
        "identity_sin_transaccion": int(only_identity),
    }


def _performance_checks(txn: pd.DataFrame) -> list[dict]:
    checks = []
    dt = txn["TransactionDT"]
    checks.append(
        {
            "check": "TransactionDT sin nulos y monotona",
            "pass": not dt.isna().any() and dt.is_monotonic_increasing,
            "detail": f"nulos={int(dt.isna().sum())}, monotona={bool(dt.is_monotonic_increasing)}",
        }
    )
    if TARGET in txn.columns:
        vals = txn[TARGET].dropna().astype(float).unique()
        checks.append(
            {
                "check": "Target binario (0/1)",
                "pass": set(vals) <= {0.0, 1.0},
                "detail": f"valores_observados={sorted(vals)[:10]}",
            }
        )
        counts = txn[TARGET].value_counts(dropna=False)
        pos = int(counts.get(1, 0))
        neg = int(counts.get(0, 0))
        checks.append(
            {
                "check": "Desbalance de clases documentado",
                "pass": 0 < pos < neg,
                "detail": f"fraude={pos} ({100*pos/len(txn):.2f}%), no_fraude={neg}",
            }
        )
    return checks


def _missing_summary(txn: pd.DataFrame, identity: pd.DataFrame) -> dict:
    out = {}
    for name, df in (("transaction", txn), ("identity", identity)):
        n = len(df)
        miss = df.isna().sum()
        out[name] = {
            "columnas_con_missing": int((miss > 0).sum()),
            "total_valores": int(np.prod(df.shape)),
            "total_missing": int(miss.sum()),
            "missing_pct_global": round(100.0 * miss.sum() / np.prod(df.shape), 2),
            "top_missing": [
                {"columna": c, "missing": int(v), "pct": round(100.0 * v / n, 2)}
                for c, v in miss.sort_values(ascending=False).head(10).items()
            ],
        }
    return out


def _expected_columns(txn: pd.DataFrame, identity: pd.DataFrame) -> list[dict]:
    families = {
        "transaction": {
            "TARGET": [TARGET],
            "clave_y_tiempo": ["TransactionID", "TransactionDT"],
            "importe": ["TransactionAmt"],
            "producto": ["ProductCD"],
            "tarjeta": [f"card{i}" for i in range(1, 7)],
            "direccion_distancia": ["addr1", "addr2", "dist1", "dist2"],
            "email": ["P_emaildomain", "R_emaildomain"],
            "m1_m9": [f"M{i}" for i in range(1, 10)],
            "c1_c14": [f"C{i}" for i in range(1, 15)],
            "d1_d15": [f"D{i}" for i in range(1, 16)],
            "v1_v339": [f"V{i}" for i in range(1, 340)],
        },
        "identity": {
            "clave": ["TransactionID"],
            "dispositivo": ["DeviceType", "DeviceInfo"],
            "id01_id38": [f"id_{i:02d}" for i in range(1, 39)],
        },
    }
    results = []
    for table, groups in families.items():
        df = txn if table == "transaction" else identity
        for group, cols in groups.items():
            present = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]
            results.append(
                {
                    "tabla": table,
                    "grupo": group,
                    "esperadas": len(cols),
                    "presentes": len(present),
                    "ausentes": missing[:20],
                    "ok": len(missing) == 0,
                }
            )
    return results


def build_report(data_dir: str | Path, report_dir: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    print("[1/7] Cargando train_transaction.csv ...")
    txn = load_transaction(data_dir)
    print("[2/7] Cargando train_identity.csv ...")
    identity = load_identity(data_dir)

    print("[3/7] Comprobaciones estructurales ...")
    structural = _structural_checks(txn, identity)
    print("[4/7] Cobertura del join ...")
    join = _join_coverage(txn, identity)
    print("[5/7] Checks de comportamiento y target ...")
    performance = _performance_checks(txn)
    print("[6/7] Resumen de missing values ...")
    missing = _missing_summary(txn, identity)
    print("[7/7] Verificacion de familias de columnas ...")
    columns = _expected_columns(txn, identity)

    report = {
        "dataset": "IEEE-CIS Fraud Detection",
        "archivos": {
            "transaction": TRANSACTION_FILE,
            "identity": IDENTITY_FILE,
        },
        "dimensiones": {
            "transaction": {"filas": int(len(txn)), "columnas": int(txn.shape[1])},
            "identity": {
                "filas": int(len(identity)),
                "columnas": int(identity.shape[1]),
            },
            "transaction_dtypes": _dtype_summary(txn),
            "identity_dtypes": _dtype_summary(identity),
        },
        "struct_checks": structural,
        "join": join,
        "performance_checks": performance,
        "missing": missing,
        "column_families": columns,
        "conclusion": all(c["pass"] for c in structural)
        and all(c["pass"] for c in performance)
        and all(c["ok"] for c in columns),
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "validation_report.json"
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Reporte guardado en {out}")

    return report
