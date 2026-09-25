# Detección de Fraude — IEEE-CIS

Sistema de Machine Learning para estimar si una transacción es fraudulenta, sobre el dataset **IEEE-CIS Fraud Detection** (Kaggle). Desarrollo incremental orientado a ML Engineering / MLOps.

## Estado actual

Fase B completada: datos IEEE-CIS incorporados y validados (entrenamiento). Ver `reports/validation_report.json` y `tasks.md`.

## Documentación

- `agents.md` — cómo trabajan los agentes.
- `constitution.md` — principios y reglas innegociables del proyecto.
- `specs.md` — requisitos y comportamiento esperado del sistema.
- `tasks.md` — plan de implementación en fases.

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

- Python 3.14
- Dependencias core en `requirements.txt`
- Entorno virtual: `.venv/` (no versionado)