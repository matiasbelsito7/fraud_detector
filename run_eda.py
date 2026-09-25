from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eda.exploration import build_eda

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    data_dir = ROOT / "data"
    out_dir = ROOT / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_eda(data_dir, out_dir)

    out_file = out_dir / "eda_results.json"
    out_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Resultados guardados en {out_file}")

    print("\n--- RECAP ---")
    print(
        f"transaction: {result['dimensiones']['transaction']['filas']}x"
        f"{result['dimensiones']['transaction']['columnas']}"
    )
    print(f"target fraude: {result['target']['fraud_pct']}%")
    m = result["missing_transaction"]
    print(
        "missing buckets txn: sin_missing={} 0-10={} 10-50={} 50-90={} 90-100={}".format(
            m["buckets"]["sin_missing"],
            m["buckets"]["0_10"],
            m["buckets"]["10_50"],
            m["buckets"]["50_90"],
            m["buckets"]["90_100"],
        )
    )
    print("columnas constantes:", result["constant_columns"])
    print("span temporal (dias):", result["time"]["dias_span"])
    print(
        "fraude por semana (min/max %): {} / {}".format(
            result["time"]["fraud_rate_min_week_pct"],
            result["time"]["fraud_rate_max_week_pct"],
        )
    )
    print("hora pico fraude:", result["time"]["hora_mayor_rate"])
    print("top corr:", [(r["columna"], r["corr"]) for r in result["corr_top20"][:5]])
    print("graficos:", result["graficas"])
