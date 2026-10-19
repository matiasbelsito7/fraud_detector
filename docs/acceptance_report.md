# Revision de aceptacion (Fase O)

Fecha: 2026-10-19. Alcance: fases obligatorias A-L contra `specs.md §15`. Las fases M (API) y N (Docker, CI/CD, monitoring) estan diferidas por `specs.md §12-13` y no bloquean esta revision.

## 1. Veredicto

Los nueve criterios de aceptacion de `specs.md §15` se cumplen y tienen evidencia verificable en el repositorio o en artefactos regenerables. El proyecto queda **completo en su alcance inicial**.

El objetivo de negocio de recall >= 0,80 **no se cumple en `test`** (0,7707) y queda documentado como riesgo abierto, no como criterio de `§15`. `test` esta consumido y bloqueado; no se reutiliza para esta revision.

## 2. Checklist de criterios

| Criterio (`specs.md §15`) | Evidencia | Estado |
|---|---|---|
| Datos validables | `run_data_validation.py` + `reports/validation_report.json`: 590.540 x 394 y 144.233 x 41; `TransactionID` int64 unico y sin nulos en ambos; join 24,42 %; 0 identity huerfanas; target 3,50 %; missing documentado. Sintesis en `docs/eda_report.md`. | Cumple |
| Preprocessing reproducible | `run_preprocessing.py` ajusta, serializa, recarga y compara SHA-256 de dos salidas (`IDENTICO: True`). Decisiones en `docs/preprocessing_report.md`; tests en `tests/test_preprocessor.py`. | Cumple |
| Features consistentes | `compute_features` es el unico modulo de features y lo usan entrenamiento e inferencia. `run_feature_engineering.py` compara SHA-256 de dos generaciones. En Fase L, la fila unica reproduce al pipeline por lotes con diferencia 0,00e+00. | Cumple |
| Sin leakage conocido | Split temporal y tests de no-leakage (`docs/split_report.md`, `docs/features_report.md`); el pipeline de inferencia rechaza historial con `TransactionDT` futuro; candado `config/test_consumed.json`. | Cumple |
| Entrenamiento reproducible | `docs/modeling_report.md`: dos ejecuciones completas de `run_training.py` dieron las mismas 6 metricas; busqueda y seleccion en `docs/tuning_report.md`. | Cumple |
| Metricas calculables | ROC-AUC, PR-AUC, recall en threshold y matriz de confusion calculadas en `docs/evaluation_report.md` y `reports/evaluation/`; curvas ROC/PR y barrido de thresholds disponibles. | Cumple |
| Experimentos registrados | MLflow con 50 runs idempotentes de G-J, commit/split/config/metricas/artefactos y tag `test_usado`; `docs/tracking_report.md`. | Cumple |
| Pipeline completo | `run_inference.py`: una transaccion real de `validation` (TransactionID 3421176) con 434.176 filas de historial atraviesa preprocessing, features, modelo y umbral sin errores. | Cumple |
| Inferencia consistente | `InferencePipeline` reutiliza `preprocessor.joblib` y `compute_features`, verifica el orden exacto de las 791 features y reproduce la probabilidad del lote (diferencia 0,00e+00); 7 tests en `tests/test_inference.py`. | Cumple |

## 3. Verificaciones ejecutadas

- `uv run ruff check` sobre `src`, `tests` y los 12 `run_*.py`: OK.
- `uv run black --check`: 51 archivos sin cambios.
- `uv run mypy` sobre `src` y los 12 scripts: 39 archivos, sin errores.
- `uv run pytest`: **121 tests passed**.
- `uv run pre-commit run --all-files`: ruff, ruff-format, black y mypy en verde.
- `uv run run_inference.py`: probabilidad 0,000324, diferencia 0,00e+00 frente al lote, `test` no cargado.

`run_evaluation.py` **no se ejecuto** en esta fase.

## 4. Hallazgos de la revision

- Se corrigio una etiqueta del EDA: el 23,84 % correspondia a cobertura de `DeviceType`, no al join con identity (24,42 %). El dato de `reports/eda/eda_results.json` era correcto; el texto lo describia mal.
- Las cifras de `validation` de T-E02 (PR-AUC 0,6079, recall 0,8002, 11,30 % de alertas) y de Fase H (0,5814, 12,87 %) coinciden entre `docs/client_features_report.md`, `docs/tuning_report.md` y `reports/tuning/`.
- No se encontraron contradicciones adicionales entre documentos y artefactos.

## 5. Riesgos y pendientes conocidos

1. **Recall de negocio:** 0,7707 en `test`, por debajo de 0,80; hay deriva temporal (0,8161 train OOF -> 0,8002 validation -> 0,7707 test). No se corrige sin datos nuevos.
2. **Modelo desplegado sin metrica de `test`:** se sirve T-E02 por su punto operativo en `validation`; su mejora no esta certificada y las cifras de `test` corresponden a Fase H. Decision y limites en `docs/inference_report.md`.
3. **`test` consumido:** `config/test_consumed.json` bloquea una segunda lectura. Cualquier reevaluacion exigiria `--force` y quedaria trazada; no debe hacerse para elegir modelos.
4. **Historial en produccion:** la inferencia es sin estado; el entorno debe aportar el historial previo del cliente. Sin historial la prediccion funciona pero degrada. Ademas, `compute_features` recalcula agregados sobre todo el historial, lo que puede requerir agregados incrementales a alto volumen.
5. **Artefactos no versionados:** `data/`, `artifacts/*.joblib`, `reports/` y `mlruns/` estan gitignored. Reproducir exige los CSV raw y ejecutar la cadena de scripts.
6. **Fases diferidas:** API (M) y Docker/CI/CD/monitoring (N) quedan fuera del alcance inicial, segun `specs.md §12-13`.
7. **Mejora adyacente no abordada:** `card1`, `addr1`, `card2`, `card3`, `addr2`, `D1`, `P_emaildomain` y `ProductCD` se tratan como numericas y no como categoricas; esta identificado en `tasks.md` y no forma parte de esta revision.

## 6. Reproduccion

```text
uv sync
uv run run_data_validation.py
uv run run_eda.py
uv run run_preprocessing.py
uv run run_feature_engineering.py
uv run run_split.py
uv run run_training.py
uv run run_tuning.py
uv run run_threshold.py
uv run run_explainability.py
uv run run_tracking.py
uv run run_inference.py
```

`run_evaluation.py` se omite a proposito: ya consumio `test` una vez y el candado impide repetirlo. Requiere `train_transaction.csv` y `train_identity.csv` en `data/raw/`.
