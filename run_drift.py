from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.inference.pipeline import Deployment, file_digest
from src.modeling.dataset import TRAIN_SET, VAL_SET, load_splits
from src.monitoring.drift import DriftMonitor, build_reference

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "data" / "processed" / "split.parquet"
DEPLOYMENT_FILE = ROOT / "config" / "deployment.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "drift_reference.json"
DEFAULT_SUMMARY = ROOT / "reports" / "drift" / "summary.json"

#: `TransactionDT` es un indice temporal monotonico: con un split temporal su
#: PSI siempre seria enorme y taparia el resto de las features. El drift de las
#: features de comportamiento ya captura el efecto temporal relevante.
NON_MONITORED = frozenset({"TransactionDT"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construye la referencia de drift desde train (nunca test) y, "
            "opcionalmente, la contrasta contra validation."
        )
    )
    parser.add_argument("--split", type=Path, default=SPLIT_FILE)
    parser.add_argument("--deployment", type=Path, default=DEPLOYMENT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--features",
        type=int,
        default=25,
        help="Numero de features de mayor ganancia que se monitorean.",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare",
        choices=("none", "validation"),
        default="validation",
        help="Conjunto para una comprobacion descriptiva del monitor.",
    )
    return parser


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _top_features(model: Any, available: set[str], n_features: int) -> list[str]:
    names = list(getattr(model, "feature_name_", []))
    if not names:
        raise ValueError("El modelo no expone feature_name_")
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("El modelo no expone booster_ para leer la ganancia")
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    order = np.argsort(gains)[::-1]
    selected = [
        names[index]
        for index in order
        if names[index] in available and names[index] not in NON_MONITORED
    ]
    if not selected:
        raise ValueError("Ninguna feature del modelo esta en el split")
    return selected[:n_features]


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.features < 1:
        raise ValueError("features debe ser >= 1")
    if args.max_rows < 1:
        raise ValueError("max_rows debe ser >= 1")
    if not args.split.exists():
        raise FileNotFoundError(f"Falta {args.split}")

    print("[1/4] Cargando split, deployment y modelo ...")
    names = (TRAIN_SET,) if args.compare == "none" else (TRAIN_SET, VAL_SET)
    splits = load_splits(args.split, names=names)
    train = splits[TRAIN_SET]
    deployment = Deployment.load(args.deployment)
    model = joblib.load(deployment.model_path)
    features = _top_features(model, set(train.X.columns), args.features)
    digest = file_digest(args.split)
    print(
        f"  train: n={train.n_rows} features={train.X.shape[1]} "
        f"monitorizadas={len(features)}"
    )

    print(f"[2/4] Construyendo referencia con {args.max_rows} filas de train ...")
    sample = (
        train.X.sample(n=args.max_rows, random_state=args.seed)
        if train.n_rows > args.max_rows
        else train.X
    )
    metadata = {
        "source": TRAIN_SET,
        "split_sha256": digest,
        "n_rows_available": int(train.n_rows),
        "n_rows_sampled": int(len(sample)),
        "seed": int(args.seed),
        "importance_type": "gain",
        "excluded_features": sorted(NON_MONITORED),
        "test_usado": False,
    }
    reference = build_reference(
        sample,
        features,
        bins=args.bins,
        metadata=metadata,
    )
    output = reference.save(args.output)
    print(f"  referencia -> {output}")

    record: dict[str, Any] = {
        "split_file": _relative(args.split),
        "split_sha256": digest,
        "test_usado": False,
        "reference": {
            "output": _relative(output),
            "features": features,
            "bins": args.bins,
            "metadata": metadata,
        },
    }

    if args.compare == "validation":
        print("[3/4] Contrastando validation contra la referencia ...")
        valid = splits[VAL_SET]
        monitor = DriftMonitor(reference, max_observations=max(valid.n_rows, 1))
        monitor.observe_frame(valid.X)
        snapshot = monitor.snapshot()
        record["validation"] = snapshot
        print(
            f"  validation: n={valid.n_rows} estado={snapshot['status']} "
            f"max_psi={snapshot['max_psi']}"
        )
        for row in snapshot["features"][:10]:
            print(f"    {row['feature']:28s} psi={row['psi']} estado={row['status']}")
    else:
        print("[3/4] Comparacion contra validation omitida ...")

    print("[4/4] Guardando resumen ...")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  resumen -> {args.summary}")


if __name__ == "__main__":
    main()
