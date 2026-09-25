# Detección de Fraude — IEEE-CIS

Sistema de Machine Learning para estimar si una transacción es fraudulenta, sobre el dataset **IEEE-CIS Fraud Detection** (Kaggle). Desarrollo incremental orientado a ML Engineering / MLOps.

## Estado actual

Fase E completada: componente de feature engineering sin información futura (tiempo, frecuencia, historial, entidades) con tests y documentación. Reporte: `docs/features_report.md`; matriz en `data/processed/train_features.parquet`.

## Documentación

- `agents.md` — cómo trabajan los agentes.
- `constitution.md` — principios y reglas innegociables del proyecto.
- `specs.md` — requisitos y comportamiento esperado del sistema.
- `tasks.md` — plan de implementación en fases.
- `docs/eda_report.md` — conclusiones del análisis exploratorio.
- `docs/preprocessing_report.md` — decisiones y reproducibilidad del preprocessing.
- `docs/features_report.md` — catálogo de features y verificación de no-leakage.

## Estructura inicial

```
data/        Datos raw (no versionados; se agregan en Fase B)
notebooks/   Análisis exploratorio (EDA)
src/         Código del proyecto (paquete Python)
config/      Configuración (a definir en fases posteriores)
tests/       Tests (se incorporan en fases posteriores)
```

Estructura provisional; puede ajustarse según las decisiones tomadas en fases siguientes (`specs.md §14`).

## Entorno

- Python 3.14, gestionado con **uv**.
- Dependencias declaradas en `pyproject.toml` (grupo de desarrollo en `[dependency-groups]`); versiones bloqueadas en `uv.lock`.
- Entorno virtual: `.venv/` (no versionado). Sincronizar con `uv sync`.
- Ejecutar scripts y herramientas con `uv run ...` (p. ej. `uv run run_eda.py`).