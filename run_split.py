from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.split.split import (
    DT_COL,
    TRAIN_WEEKS,
    VAL_WEEKS,
    apply_split,
    check_summary,
    summary,
    week_index,
)

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_FILE = "train_features.parquet"
SPLIT_FILE = "split.parquet"


def main() -> None:
    print("[1/4] Cargando matriz de features ...")
    features_file = PROCESSED_DIR / FEATURES_FILE
    df = pd.read_parquet(features_file)
    print(f"  {df.shape} | columnas={df.shape[1]}")

    print("[2/4] Aplicando split temporal ...")
    split_df = apply_split(df)
    weeks_total = int(week_index(split_df[DT_COL]).max()) + 1
    print(
        f"  semanas observadas={weeks_total} "
        f"(train={TRAIN_WEEKS} val={VAL_WEEKS} test={weeks_total - TRAIN_WEEKS - VAL_WEEKS})"
    )

    print("[3/4] Validacion del split ...")
    checks = check_summary(split_df)
    for c in checks:
        print(f"  {'OK ' if c['pass'] else 'FALLA'} | {c['check']} | {c['detail']}")
    assert all(c["pass"] for c in checks), "La validacion del split fallo"

    print("[4/4] Resumen y guardado ...")
    print(summary(split_df).to_string(index=False))
    out_file = PROCESSED_DIR / SPLIT_FILE
    split_df.to_parquet(out_file, index=False)
    print(f"  guardado -> {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
