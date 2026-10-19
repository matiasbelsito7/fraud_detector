# Detección de Fraude — IEEE-CIS

Sistema de Machine Learning para estimar si una transacción es fraudulenta, sobre el dataset **IEEE-CIS Fraud Detection** (Kaggle). Desarrollo incremental orientado a ML Engineering / MLOps.

## Estado actual

Fase I completada. El modelo final de la Fase H (LightGBM, configuración 18) tiene un umbral operativo de `0,029329`, fijado en `validation` **antes** de leer `test`. Consumido una sola vez, `test` da ROC-AUC 0,9025, PR-AUC 0,5282 y recall 0,7707 [0,7551, 0,7864] con 15,94 % de alertas.

**El objetivo de negocio de recall ≥ 0,80 no se cumple en `test`**: hay deriva temporal (recall 0,8161 → 0,8002 → 0,7707 en train OOF, validation y test). `test` ya no puede volver a usarse para decidir nada. Reporte: `docs/evaluation_report.md`; resultados en `reports/evaluation/`.

> **Las cifras anteriores son del modelo sin features de identidad de cliente.** T-E02 se implementó después (2026-09-28) y **no se evaluó sobre `test`**: al ser selección de features, `constitution.md` §5 exige que `test` no participe de ella. Su modelo se certificó solo con `train` out-of-fold y `validation`, con resultado **negativo**: la comparación aislada da −0,0016 de PR-AUC medio con los tres pliegues de distinto signo, dentro del ruido de ±0,05. El único punto a favor es el operativo (mismo recall con 11,30 % de alertas frente a 12,87 %). La Fase L fija **T-E02 como modelo desplegable** por ese punto operativo en `validation`, no por una mejora certificada; las cifras de `test` siguen correspondiendo al modelo sin features de cliente. Análisis completo en `docs/client_features_report.md`.

Fase J completada (2026-10-18) sobre el artefacto hoy seleccionado (`model_final.joblib`, config 11): importancia global y SHAP con TreeSHAP nativo de LightGBM, sin cargar `test`. `V258` concentra la mayor ganancia (10,25 %); por `|SHAP|` encabezan `C13`, `since_first_sec_card1` y `TransactionAmt`. Las 29 features de cliente aportan 15,48 % del `|SHAP|`, pero eso no certifica su mejora. Reporte: `docs/explainability_report.md`; artefactos en `reports/explainability/` (gitignored).

Fase K completada (2026-10-19): los experimentos de las Fases G–J quedaron registrados en MLflow mediante un **backfill** que lee los reportes ya existentes, sin reentrenar y **sin volver a leer `test`**. Store SQLite local en `mlruns/mlflow.db` (gitignored), 50 runs idempotentes por tag. La Fase I es el único run con `test_usado=true`. Reporte: `docs/tracking_report.md`; ejecución: `uv run run_tracking.py`.

Fase L completada (2026-10-19): pipeline de inferencia en `src/inference/pipeline.py` y configuración desplegable en `config/deployment.json` (T-E02, umbral `0,008480`, digests de artefactos). Recibe una transacción con historial opcional, aplica el mismo preprocessing y feature engineering, verifica el orden de las 791 features y devuelve probabilidad y clase. La prueba real con 434.176 filas de historial dio probabilidad idéntica al pipeline por lotes (diferencia 0,00e+00), sin cargar `test`. Reporte: `docs/inference_report.md`; demostración: `uv run run_inference.py`.

Fase O completada (2026-10-19): revisión de los 9 criterios de `specs.md §15`, todos cumplidos con evidencia verificable. El alcance inicial (fases A–L) queda cerrado. Checklist completo y riesgos abiertos en `docs/acceptance_report.md`.

Fase M completada (2026-10-19): API FastAPI en `src/api/` y `run_api.py`, con `GET /health` y `POST /predict` (transacción con historial e identidad opcionales, umbral configurable), carga del modelo al iniciar y validación de digests. 9 tests HTTP y smoke real con T-E02. Reporte: `docs/api_report.md`; ejecución: `uv run run_api.py`.

Fase N completada (2026-10-19): `Dockerfile` + `docker-compose.yml` (artefactos montados read-only, healthcheck, usuario no root), CI en GitHub Actions (calidad + build de imagen) y monitoring Prometheus en `GET /metrics`. El build local no pudo completarse por falta de espacio en disco del host; el job `docker` de CI lo valida en Linux. Reporte: `docs/operations_report.md`.

## Documentación

- `agents.md` — cómo trabajan los agentes.
- `constitution.md` — principios y reglas innegociables del proyecto.
- `specs.md` — requisitos y comportamiento esperado del sistema.
- `tasks.md` — plan de implementación en fases.
- `docs/eda_report.md` — conclusiones del análisis exploratorio.
- `docs/preprocessing_report.md` — decisiones y reproducibilidad del preprocessing.
- `docs/features_report.md` — catálogo de features y verificación de no-leakage.
- `docs/split_report.md` — estrategia de división temporal y aislamiento del test.
- `docs/modeling_report.md` — escalera de modelos, comparación y reproducibilidad del entrenamiento.
- `docs/tuning_report.md` — protocolo de búsqueda, selección del modelo final y riesgos abiertos.
- `docs/evaluation_report.md` — fijación del umbral, resultado en `test` y conclusión sobre el objetivo de recall.
- `docs/client_features_report.md` — identidad de cliente (T-E02): análisis, implementación, medición y veredicto negativo.
- `docs/explainability_report.md` — importancia global, SHAP por familia y explicaciones individuales (Fase J).
- `docs/tracking_report.md` — registro de experimentos G–J en MLflow: store, inventario de runs, trazabilidad y límites (Fase K).
- `docs/inference_report.md` — pipeline de inferencia: contrato de entrada, transformaciones, configuración desplegable, verificación y límites (Fase L).
- `docs/acceptance_report.md` — checklist de los criterios de `specs.md §15`, verificaciones ejecutadas y riesgos abiertos (Fase O).
- `docs/api_report.md` — API FastAPI: contrato, carga del modelo, errores, ejecución y límites (Fase M).
- `docs/operations_report.md` — Docker, CI/CD, tests automatizados y monitoring Prometheus (Fase N).

## Pipeline de ejecución

Los scripts se ejecutan en orden, cada uno consumiendo la salida del anterior:

```
uv run run_data_validation.py
uv run run_eda.py
uv run run_preprocessing.py
uv run run_feature_engineering.py
uv run run_split.py
uv run run_training.py
uv run run_tuning.py
uv run run_threshold.py
uv run run_evaluation.py
uv run run_explainability.py
uv run run_tracking.py
uv run run_inference.py
```

`run_threshold.py` fija el umbral y **no carga `test`**. `run_evaluation.py` es el único script que lo lee, y aborta si ya figura consumido, si el umbral no está registrado, si la config no coincide o si el digest de los datos cambió respecto al registrado en `threshold.json`. Ya se ejecutó una vez (Fase I) y **no debe volver a ejecutarse**. La comprobación de digest **ya no protege** por sí sola, porque `run_threshold.py` se re-ejecutó para T-E02 y regeneró `threshold.json` con el digest vigente; el freno efectivo es ahora el candado `config/test_consumed.json`, que registra cada consumo de `test` con independencia de los datos y bloquea la ejecución salvo `--force` explícito (que deja traza). El registro de la Fase I está en `reports/evaluation/fase_i_baseline/test_metrics.json`.

`run_explainability.py` **no carga `test`** y no necesita la salida de `run_evaluation.py`: se apoya en `model_final.joblib` y `threshold.json`, y explica `validation` con TreeSHAP nativo de LightGBM.

`run_tracking.py` registra en MLflow los resultados de las Fases G–J leyendo `reports/` y `docs/`; **no reentrena ni carga `test`**. Es idempotente (salvo `--force`) y su store por defecto es `mlruns/mlflow.db` (gitignored), por lo que no versiona datos.

`run_inference.py` demuestra el pipeline completo con una fila de `validation` y su historial previo; **no carga `test`**. Usa `config/deployment.json` y exige que los artefactos coincidan con sus sha256. Para servir una transacción nueva, usar `InferencePipeline.load(...).predict(...)` con el historial del cliente.

La API se levanta aparte de la cadena batch: `uv run run_api.py` (por defecto `127.0.0.1:8000`). Expone `GET /health`, `POST /predict` y `GET /metrics` (Prometheus), y documenta el contrato en `docs/api_report.md`. No carga `test`. También puede ejecutarse en Docker: `docker compose up --build` (requiere `artifacts/`).

## Estructura inicial

```
artifacts/  Modelos serializados (no versionados)
data/       Datos raw y derivados (no versionados)
notebooks/  Análisis exploratorio (EDA)
reports/    Resultados de las fases (no versionados)
src/        Código del proyecto (paquete Python)
config/     Configuración desplegable (deployment.json, candado de test)
tests/      Tests automatizados
```

Estructura provisional; puede ajustarse según las decisiones tomadas en fases siguientes (`specs.md §14`).

## Entorno

- Python 3.14, gestionado con **uv**.
- Dependencias declaradas en `pyproject.toml` (grupo de desarrollo en `[dependency-groups]`); versiones bloqueadas en `uv.lock`.
- Entorno virtual: `.venv/` (no versionado). Sincronizar con `uv sync`.
- Experiment tracking con **MLflow** (Fase K); el store vive en `mlruns/` (no versionado).
- API con **FastAPI** y **Uvicorn** (Fase M); tests HTTP con `httpx`.
- CI en **GitHub Actions** y contenedor **Docker** (Fase N); metricas con `prometheus-client`.
- Ejecutar scripts y herramientas con `uv run ...` (p. ej. `uv run run_eda.py`).