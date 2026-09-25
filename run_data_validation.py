from __future__ import annotations

import sys
from pathlib import Path

from src.data.validation import build_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    data_dir = ROOT / "data"
    report_dir = ROOT / "reports"
    report = build_report(data_dir, report_dir)

    print("\n--- RESUMEN AREA STATUS ---")
    for c in report["struct_checks"]:
        print(f"{'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    for c in report["performance_checks"]:
        print(f"{'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    print(
        f"JOIN | coverage={report['join']['coverage_pct']}% "
        f"({report['join']['transacciones_con_identity']}/{report['join']['transacciones']})"
    )
    print(
        "COLUMNAS | transaction con familias ausentes: "
        f"{sum(1 for g in report['column_families'] if g['tabla'] == 'transaction' and not g['ok'])}, "
        f"identity: {sum(1 for g in report['column_families'] if g['tabla'] == 'identity' and not g['ok'])}"
    )
    print(f"CONCLUSION GLOBAL: {'VALIDADO' if report['conclusion'] else 'REVISAR'}")
