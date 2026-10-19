"""Explainability del modelo final: importancia global y explicaciones individuales.

No consume `test`: trabaja sobre el artefacto ya seleccionado y sobre
`validation`, el unico bloque fuera de muestra. `train` no hace falta para SHAP
y se evita cargarlo. El objetivo de la Fase J es entender el modelo, no medirlo
ni elegir nada, asi que ni `test` ni `validation` se usan para tomar decisiones.

Salidas (gitignored) en `reports/explainability/`:

- `importance_gain.csv` / `importance_gain_by_family.csv`
- `shap_global.csv` / `shap_by_family.csv`
- `summary.json` (metadata, top features, ejemplos individuales)

La tecnica SHAP es el TreeSHAP nativo de LightGBM (`pred_contrib=True`). No se
anade la dependencia `shap`: daria los mismos valores exactos y no aporta nada
sobre esta implementacion (`constitution.md` §7, `specs.md` §9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.explainability.importance import (
    aggregate_by_family,
    check_additivity,
    gain_importance,
    local_contributions,
    shap_base_value,
    shap_contributions,
    shap_summary,
)
from src.modeling.dataset import TEST_SET, VAL_SET, load_splits
from src.modeling.models import SEED

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
EXPLAIN_REPORTS = REPORTS_DIR / "explainability"
TUNING_DIR = REPORTS_DIR / "tuning"
EVAL_REPORTS = REPORTS_DIR / "evaluation"

SPLIT_FILE = "split.parquet"
MODEL_FILE = ROOT / "artifacts" / "model_final.joblib"
SELECTED_JSON = TUNING_DIR / "selected.json"
THRESHOLD_JSON = EVAL_REPORTS / "threshold.json"

SAMPLE_SIZE = 5_000
TOP_FEATURES = 20
TOP_LOCAL = 10

GAIN_CSV = "importance_gain.csv"
GAIN_FAMILY_CSV = "importance_gain_by_family.csv"
SHAP_CSV = "shap_global.csv"
SHAP_FAMILY_CSV = "shap_by_family.csv"
SUMMARY_JSON = "summary.json"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    """Muestra determinista de posiciones, ordenada para reproducir el orden."""
    size = min(int(sample_size), n_rows)
    if size >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _individual_examples(
    estimator: Any,
    valid: Any,
    feature_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Explica un TP, un FN y un FP en el umbral operativo ya registrado.

    Se elige la transaccion de cada grupo con mayor score: el TP mas confiado,
    el FN mas cerca de ser detectado y el FP mas confiado. Son casos
    deterministas, extremos y faciles de interpretar; no pretenden ser una
    muestra representativa.
    """
    y = np.asarray(valid.y).astype(int)
    scores = np.asarray(estimator.predict_proba(valid.X))[:, 1]
    predicted = scores >= threshold
    groups = {
        "verdadero_positivo": np.flatnonzero((y == 1) & predicted),
        "falso_negativo": np.flatnonzero((y == 1) & ~predicted),
        "falso_positivo": np.flatnonzero((y == 0) & predicted),
    }
    out: dict[str, Any] = {}
    for label, positions in groups.items():
        if positions.size == 0:
            out[label] = {"n": 0}
            continue
        chosen = int(positions[np.argmax(scores[positions])])
        explanation = local_contributions(
            estimator, valid.X.iloc[[chosen]], feature_names, top=TOP_LOCAL
        )
        out[label] = {
            "n": int(positions.size),
            "fila_split": int(valid.X.index[chosen]),
            "score": float(scores[chosen]),
            "umbral": float(threshold),
            "prediccion": bool(predicted[chosen]),
            "etiqueta": int(y[chosen]),
            **explanation,
        }
    return out


def main(sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> None:
    split_file = PROCESSED_DIR / SPLIT_FILE
    if not split_file.exists():
        raise FileNotFoundError(
            f"Falta {split_file}. Ejecuta antes el pipeline hasta run_split.py"
        )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Falta {MODEL_FILE}. Ejecuta antes: run_tuning.py")
    if not SELECTED_JSON.exists():
        raise FileNotFoundError(f"Falta {SELECTED_JSON}. Ejecuta antes: run_tuning.py")
    if not THRESHOLD_JSON.exists():
        raise FileNotFoundError(
            f"Falta {THRESHOLD_JSON}. Ejecuta antes: run_threshold.py"
        )

    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    threshold_record = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    if selected.get("test_usado") is not False:
        raise ValueError("selected.json no certifica test_usado=false")
    if threshold_record.get("test_usado") is not False:
        raise ValueError("threshold.json no certifica test_usado=false")
    if threshold_record["config_id"] != selected["seleccion"]["config_id"]:
        raise ValueError("El modelo seleccionado y el umbral no son de la misma config")

    digest = _file_digest(split_file)
    if digest != selected["split_sha256"]:
        raise ValueError(
            "split.parquet cambio despues del tuning "
            f"({digest[:16]} != {selected['split_sha256'][:16]}): el artefacto "
            "no corresponde a estos datos. Re-ejecuta run_tuning.py"
        )

    print("[1/6] Cargando validation (test NO se carga) ...")
    splits = load_splits(split_file, names=(VAL_SET,))
    assert TEST_SET not in splits, "Explainability no debe cargar test"
    valid = splits[VAL_SET]
    feature_names = list(valid.X.columns)
    print(
        f"  validation: n={valid.n_rows} features={len(feature_names)} "
        f"fraude={100 * valid.pos_rate:.3f}%"
    )

    estimator = joblib.load(MODEL_FILE)
    if int(estimator.n_features_in_) != len(feature_names):
        raise ValueError(
            f"El modelo espera {estimator.n_features_in_} features y validation "
            f"tiene {len(feature_names)}"
        )
    threshold = float(threshold_record["threshold"])
    print(
        f"  modelo: {type(estimator).__name__}, config "
        f"{selected['seleccion']['config_id']}, {estimator.n_estimators} rondas"
    )

    print("[2/6] Importancia global por ganancia del booster ...")
    gain = gain_importance(estimator, feature_names)
    gain_family = aggregate_by_family(gain, "gain_pct")
    print(
        f"  top-1 gain: {gain.iloc[0]['feature']} "
        f"({gain.iloc[0]['gain_pct']:.2f}% de la ganancia total)"
    )

    print(
        f"[3/6] SHAP (TreeSHAP nativo) sobre muestra de validation "
        f"(n={min(sample_size, valid.n_rows)}, seed={seed}) ..."
    )
    positions = _sample_indices(valid.n_rows, sample_size, seed)
    sample = valid.X.iloc[positions]
    contributions = shap_contributions(estimator, sample)
    additivity = check_additivity(estimator, sample)
    if additivity > 1e-4:
        raise ValueError(
            f"Los valores SHAP no reconstruyen el score crudo (error {additivity:.2e}): "
            "la tecnica no corresponde al estimador"
        )
    print(
        f"  aditividad: error maximo |bias + suma(SHAP) - raw_score| = {additivity:.2e}"
    )

    print("[4/6] Agregando importancia SHAP global y por familia ...")
    shap = shap_summary(estimator, sample, feature_names)
    shap_family = aggregate_by_family(shap, "mean_abs_shap")
    print(
        f"  top-1 SHAP: {shap.iloc[0]['feature']} "
        f"(|SHAP| medio={shap.iloc[0]['mean_abs_shap']:.4f})"
    )

    print(f"[5/6] Explicando casos individuales en umbral {threshold:.6f} ...")
    examples = _individual_examples(estimator, valid, feature_names, threshold)

    print("[6/6] Guardando artefactos ...")
    EXPLAIN_REPORTS.mkdir(parents=True, exist_ok=True)
    gain.to_csv(EXPLAIN_REPORTS / GAIN_CSV, index=False)
    gain_family.to_csv(EXPLAIN_REPORTS / GAIN_FAMILY_CSV, index=False)
    shap.to_csv(EXPLAIN_REPORTS / SHAP_CSV, index=False)
    shap_family.to_csv(EXPLAIN_REPORTS / SHAP_FAMILY_CSV, index=False)

    record: dict[str, Any] = {
        "fase": "J (T-J01)",
        "conjunto_explicado": VAL_SET,
        "test_usado": False,
        "modelo": {
            "artefacto": MODEL_FILE.name,
            "clase": type(estimator).__name__,
            "config_id": selected["seleccion"]["config_id"],
            "n_rondas": int(estimator.n_estimators),
            "n_features": len(feature_names),
        },
        "split_sha256": digest,
        "umbral_operativo": threshold,
        "muestra_shap": {
            "n": int(len(positions)),
            "seed": int(seed),
            "metodo": "muestreo uniforme sin reemplazo sobre validation",
        },
        "shap": {
            "tecnica": "TreeSHAP nativo de LightGBM (predict(pred_contrib=True))",
            "base_value_medio": shap_base_value(contributions),
            "error_aditividad_max": additivity,
        },
        "familias": gain.groupby("family")
        .size()
        .reset_index(name="n_features")
        .to_dict(orient="records"),
        "top_gain": gain.head(TOP_FEATURES)[
            ["rank", "feature", "gain_pct", "family"]
        ].to_dict(orient="records"),
        "top_shap": shap.head(TOP_FEATURES)[
            ["rank", "feature", "mean_abs_shap", "mean_shap", "family"]
        ].to_dict(orient="records"),
        "gain_por_familia": gain_family.to_dict(orient="records"),
        "shap_por_familia": shap_family.to_dict(orient="records"),
        "ejemplos_individuales": examples,
        "modelo_pendiente_de_decision": {
            "nota": (
                "T-E02 no esta certificado como mejora. Existe tambien "
                "artifacts/model_final_fase_h.joblib con metricas de test; la "
                "decision de cual desplegar sigue abierta y esta explicabilidad "
                "describe solo el artefacto seleccionado por run_tuning.py."
            )
        },
    }
    path = EXPLAIN_REPORTS / SUMMARY_JSON
    path.write_text(
        json.dumps(_json_ready(record), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  gain -> {EXPLAIN_REPORTS / GAIN_CSV}")
    print(f"  gain por familia -> {EXPLAIN_REPORTS / GAIN_FAMILY_CSV}")
    print(f"  SHAP global -> {EXPLAIN_REPORTS / SHAP_CSV}")
    print(f"  SHAP por familia -> {EXPLAIN_REPORTS / SHAP_FAMILY_CSV}")
    print(f"  resumen -> {path}")

    print("\n  Top-10 importancia SHAP (media |SHAP|):")
    for row in shap.head(10).to_dict(orient="records"):
        print(
            f"    {row['rank']:>3}. {row['feature']:<32} "
            f"|SHAP|={row['mean_abs_shap']:.4f}  familia={row['family']}"
        )
    print("\n  Familias por |SHAP| acumulado:")
    for row in shap_family.head(8).to_dict(orient="records"):
        print(
            f"    {row['family']:<26} n={row['n_features']:>4} "
            f"{row['mean_abs_shap']:>9.4f} ({row['pct']:>5.2f}%)"
        )
    print("\n  Casos individuales:")
    for label, example in examples.items():
        if not example.get("n"):
            print(f"    {label}: sin casos en validation")
            continue
        print(
            f"    {label}: fila={example['fila_split']} score={example['score']:.4f} "
            f"etiqueta={example['etiqueta']}"
        )
        for item in example["top"][:5]:
            print(
                f"      {item['feature']:<32} value={item['value']:>10.3f} "
                f"SHAP={item['shap']:+.4f}"
            )
    print("\n  test no se uso. Explainability queda asociada al experimento.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Filas de validation para SHAP (por defecto {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Semilla del muestreo (por defecto {SEED})",
    )
    args = parser.parse_args()
    main(sample_size=args.sample_size, seed=args.seed)
