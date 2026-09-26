# Detección de Fraude — IEEE-CIS

Sistema de Machine Learning para estimar si una transacción es fraudulenta, sobre el dataset **IEEE-CIS Fraud Detection** (Kaggle). Desarrollo incremental orientado a ML Engineering / MLOps.

## Estado actual

Fase H completada: búsqueda de 20 configuraciones de LightGBM sobre 3 pliegues walk-forward internos de `train`, con la configuración de la Fase G como ancla. El modelo definitivo quedó ajustado sobre `train` y confirmado **una sola vez** en `validation`; el `test` permanece intacto y se consumirá en la Fase I. Reporte: `docs/tuning_report.md`; resultados en `reports/tuning/`, modelo en `artifacts/model_final.joblib`.

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
```

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