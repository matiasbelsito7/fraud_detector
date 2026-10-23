from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.modeling.dataset import TRAIN_SET, load_splits
from src.modeling.folds import check_folds, walk_forward_folds
from src.modeling.robustness import evaluate_repeated
from src.preprocessing.categoricals import CATEGORICAL_ENCODINGS

ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "data" / "processed" / "split.parquet"
SELECTED_FILE = ROOT / "reports" / "tuning" / "selected.json"
DEFAULT_OUTPUT = ROOT / "reports" / "ablation" / "summary.json"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "E0/E1: compara encoding ordinal vs categorico nativo con la "
            "configuracion ganadora, varios seeds y presupuestos de alerta."
        )
    )
    parser.add_argument("--split", type=Path, default=SPLIT_FILE)
    parser.add_argument("--selected", type=Path, default=SELECTED_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=CATEGORICAL_ENCODINGS,
        default=list(CATEGORICAL_ENCODINGS),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Remuestras bootstrap por seed; 0 desactiva los intervalos.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Sobrescribe las rondas guardadas en selected.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.split.exists():
        raise FileNotFoundError(
            f"Falta {args.split}. Ejecuta antes el pipeline de datos."
        )
    if not args.selected.exists():
        raise FileNotFoundError(f"Falta {args.selected}")

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    selection = selected["seleccion"]
    params = selection["params"]
    rounds = int(
        args.rounds if args.rounds is not None else selection["best_iteration"]
    )

    print("[1/4] Cargando solo train (validation y test NO se cargan) ...")
    splits = load_splits(args.split, names=(TRAIN_SET,))
    train = splits[TRAIN_SET]
    digest = _file_digest(args.split)
    print(
        f"  train: n={train.n_rows} features={train.X.shape[1]} "
        f"fraude={100 * train.pos_rate:.3f}%"
    )

    print(
        f"[2/4] Construyendo {args.folds} pliegues walk-forward "
        f"({args.val_weeks} semanas de validacion) ..."
    )
    folds = walk_forward_folds(
        train.X["TransactionDT"], n_folds=args.folds, val_weeks=args.val_weeks
    )
    checks = check_folds(train.X["TransactionDT"], folds)
    for check in checks:
        print(f"  {'OK ' if check['pass'] else 'FALLA'} | {check['check']}")
    if not all(check["pass"] for check in checks):
        raise RuntimeError("Los pliegues no cumplen las invariantes temporales")

    print(
        f"[3/4] Evaluando {args.encodings} con seeds={args.seeds} y {rounds} rondas ..."
    )
    results = {
        encoding: evaluate_repeated(
            params,
            rounds,
            train,
            folds,
            tuple(args.seeds),
            encoding=encoding,
            n_boot=args.bootstrap,
        )
        for encoding in args.encodings
    }

    print("[4/4] Guardando resumen ...")
    record: dict[str, Any] = {
        "split_file": str(args.split.relative_to(ROOT)),
        "split_sha256": digest,
        "selected_file": str(args.selected.relative_to(ROOT)),
        "params": params,
        "rounds": rounds,
        "folds": args.folds,
        "val_weeks": args.val_weeks,
        "seeds": args.seeds,
        "test_usado": False,
        "entrenado_y_evaluado_en": TRAIN_SET,
        "encodings": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for encoding, result in results.items():
        aggregate = result["aggregate"]
        pr_auc_mean = aggregate["pr_auc"]["mean"]
        pr_auc_std = aggregate["pr_auc"]["std"]
        line = f"  {encoding:8s} pr_auc={pr_auc_mean:.4f}+-{pr_auc_std:.4f}"
        for rate in result["alert_rates"]:
            label = f"{float(rate):.2f}"
            line += (
                f" recall@{label}={aggregate[f'recall_at_{label}']['mean']:.4f}"
                f" alertas={aggregate[f'alert_rate_at_{label}']['mean']:.4f}"
            )
        print(line)
    print(f"  resumen -> {args.output}")


if __name__ == "__main__":
    main()
