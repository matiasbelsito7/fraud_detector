"""Registra en MLflow los experimentos de las Fases G-J (T-K01).

No reentrena ni vuelve a leer `test`: lee los reportes que cada fase ya dejo en
`reports/` y `docs/` y los registra como runs, con trazabilidad a codigo
(`git_commit`), datos (`split_sha256`) y configuracion (`config_id`, rondas).

El store por defecto es `mlruns/mlflow.db` (SQLite, gitignored). Un backfill es
idempotente: re-ejecutarlo no duplica runs salvo `--force`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.tracking.backfill import BACKFILLS, ReportsLayout, run_backfill
from src.tracking.client import Tracker, configure_tracking, default_tracking_uri

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "URI de tracking de MLflow. Por defecto "
            f"{default_tracking_uri(ROOT)} (SQLite bajo mlruns/)."
        ),
    )
    parser.add_argument("--experiment", default="fraud_detector")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(BACKFILLS),
        default=None,
        help="Fases a registrar (por defecto, todas).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Duplica runs aunque ya exista su clave de tracking.",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="No registra el estimador en formato MLflow (solo params/metricas/artefactos).",
    )
    args = parser.parse_args(argv)

    tracking_uri = args.tracking_uri or default_tracking_uri(ROOT)
    client, experiment_id = configure_tracking(tracking_uri, args.experiment)
    tracker = Tracker(
        client=client,
        experiment_id=experiment_id,
        root=ROOT,
        force=args.force,
        include_models=not args.no_models,
    )
    layout = ReportsLayout.default(ROOT)

    print(f"[1/3] Store MLflow : {tracking_uri}")
    print(f"      experimento  : {args.experiment} (id {experiment_id})")
    print(
        f"      modelos      : {'no' if args.no_models else 'si'} | force: {args.force}"
    )

    print("[2/3] Registrando fases (test NO se lee) ...")
    results = run_backfill(tracker, layout, only=args.only)
    for key, run_ids in results.items():
        label = BACKFILLS[key][0]
        print(f"  {key:4s} {label:38s} runs={len(run_ids)}")

    print("[3/3] Resumen ...")
    print(f"  runs creados : {len(tracker.created)}")
    print(f"  runs omitidos: {len(tracker.skipped)} (ya existian)")
    if tracker.missing:
        print(f"  artefactos ausentes: {len(tracker.missing)}")
        for path in tracker.missing:
            print(f"    - {path}")
    print(
        "\n  Trazabilidad: cada run lleva git_commit de la fase, split_sha256 y "
        "tag test_usado.\n  Fase I queda marcada test_usado=true (unico consumo "
        "de test, con el modelo de Fase H)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
