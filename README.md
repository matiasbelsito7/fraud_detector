# Detección de Fraude — IEEE-CIS

Sistema de Machine Learning para estimar si una transacción es fraudulenta, sobre el dataset **IEEE-CIS Fraud Detection** (Kaggle). Desarrollo incremental orientado a ML Engineering / MLOps.

## Estado actual

Fase I completada. El modelo final de la Fase H (LightGBM, configuración 18) tiene un umbral operativo de `0,029329`, fijado en `validation` **antes** de leer `test`. Consumido una sola vez, `test` da ROC-AUC 0,9025, PR-AUC 0,5282 y recall 0,7707 [0,7551, 0,7864] con 15,94 % de alertas.

**El objetivo de negocio de recall ≥ 0,80 no se cumple en `test`**: hay deriva temporal (recall 0,8161 → 0,8002 → 0,7707 en train OOF, validation y test). `test` ya no puede volver a usarse para decidir nada. Reporte: `docs/evaluation_report.md`; resultados en `reports/evaluation/`.

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
```

`run_threshold.py` fija el umbral y **no carga `test`**. `run_evaluation.py` es el único script que lo lee, y aborta si el umbral aún no está registrado o si el digest de los datos cambió.

## Estructura inicial

```
artifacts/  Modelos serializados (no versionados)
data/       Datos raw y derivados (no versionados)
notebooks/  Análisis exploratorio (EDA)
reports/    Resultados de las fases (no versionados)
src/        Código del proyecto (paquete Python)
config/     Configuración (a definir en fases posteriores)
tests/      Tests automatizados
```

Estructura provisional; puede ajustarse según las decisiones tomadas en fases siguientes (`specs.md §14`).

## Entorno

- Python 3.14, gestionado con **uv**.
- Dependencias declaradas en `pyproject.toml` (grupo de desarrollo en `[dependency-groups]`); versiones bloqueadas en `uv.lock`.
- Entorno virtual: `.venv/` (no versionado). Sincronizar con `uv sync`.
- Ejecutar scripts y herramientas con `uv run ...` (p. ej. `uv run run_eda.py`).