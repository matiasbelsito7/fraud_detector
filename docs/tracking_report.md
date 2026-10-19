# Tracking de experimentos con MLflow (Fase K)

Fecha: 2026-10-19. Modulos: `src/tracking/{client,backfill}.py`; ejecucion: `run_tracking.py`. Store por defecto: `mlruns/mlflow.db` (SQLite, gitignored).

## 1. Alcance y decision de diseno

`specs.md` §10 pide integrar MLflow para registrar experimentos, parametros, metricas, modelos y artefactos relevantes, y que los experimentos puedan reproducirse y compararse. Las Fases G–J ya dejaron sus resultados en `reports/` (gitignored) y `docs/`.

Se eligio **backfill**: `run_tracking.py` lee esos reportes y los registra como runs, sin reentrenar. Motivos:

- el tuning tarda ~87 min por ejecucion y ya se corrio dos veces (Fase H y T-E02);
- la Fase I no puede re-ejecutarse: `test` se consumio una sola vez y el candado `config/test_consumed.json` lo impide (`constitution.md` §5);
- registrar en vivo habria exigido volver a producir los mismos numeros, con riesgo de divergencia, para un resultado que ya esta medido.

La instrumentacion en vivo queda como mejora futura: para ejecuciones nuevas de Fases G–J bastaria llamar a los mismos helpers al terminar cada script.

## 2. Store y uso

- Backend **SQLite** (`sqlite:///mlruns/mlflow.db`). MLflow 3.x puso el backend de sistema de archivos en modo mantenimiento; SQLite no requiere servicio y mantiene todo en un archivo dentro de `mlruns/`, ya gitignored.
- Ejecucion completa:
  ```
  uv run run_tracking.py
  ```
- Opciones: `--tracking-uri` (URI alternativo), `--experiment` (por defecto `fraud_detector`), `--only g h e02 i j`, `--force` (duplica aunque exista la clave), `--no-models` (omite el registro del estimador en formato MLflow).
- **Idempotente:** cada run lleva un tag `tracking_key`; si ya existe, se omite. Verificado: una segunda ejecucion creo 0 runs y omitio 50.

## 3. Inventario de runs

| Fase | Clave | Runs | Que registra |
|---|---|---:|---|
| G | `fase_g`, `fase_g_<modelo>` | 6 | Padre + un run por modelo de `reports/modeling/comparison.json` (5 modelos), `comparison.{csv,json}`, `modeling_report.md` |
| H | `fase_h_seleccion`, `fase_h_config_<id>` | 21 | Padre (config ganadora, validacion, modelo `model_final_fase_h.joblib` en flavor LightGBM) + 20 configs de `baseline_fase_h/search.json` con metricas por pliegue |
| T-E02 | `te02_seleccion`, `te02_config_<id>` | 21 | Padre (config 11, umbral operativo y calibracion y corroboracion desde `threshold.json`, modelo `model_final.joblib`) + 20 configs |
| I | `fase_i_test` | 1 | Unico consumo de `test`: metricas, IC de recall, curvas y barrido de `reports/evaluation/fase_i_baseline/`, `evaluation_report.md` |
| J | `fase_j_explainability` | 1 | Muestra SHAP, base value, error de aditividad, importancias y `explainability_report.md` |

Total: **50 runs**, todos `FINISHED` (10 padres/selecciones y 40 configs de modelo).

## 4. Trazabilidad

Cada run queda enlazado a codigo, datos y configuracion (`constitution.md` §9):

| Fase | `git_commit` | `split_sha256` | `test_usado` |
|---|---|---|---|
| G | `8ab88de` | `6d612a6a…` | false |
| H | `839825c` | `6d612a6a…` | false |
| T-E02 | `95370f8` | `162c566a…` | false |
| I | `069adc6` | `6d612a6a…` | **true** |
| J | `ccaf3b7` | `162c566a…` | false |

Ademas, cada run guarda `config_id`, rondas y parametros de la configuracion cuando aplica. Los commits son los que produjeron cada experimento, no el de esta fase.

**Fase I queda marcada `test_usado=true`** y su tag `advertencia` recuerda que el modelo es el de Fase H sin features de identidad de cliente. Es el unico run con metricas de `test`; ninguna otra fase los tiene.

## 5. Limites y riesgos

- **Backfill, no logging en vivo.** El store refleja lo que los scripts ya escribieron; si un reporte se regenera, hay que re-ejecutar `run_tracking.py --force` para duplicar/actualizar.
- **Depende de `reports/` y `docs/`.** Son gitignored: un clon limpio no tiene esos artefactos y el backfill fallaria hasta re-ejecutar el pipeline. `run_tracking.py` informa los artefactos ausentes en lugar de fallar en silencio.
- **Modelos en formato MLflow (flavor LightGBM) + el `.joblib` referenciado por ruta.** El registro de modelo usa `mlflow.lightgbm.save_model`; no se uso el Model Registry remoto (`specs.md` §13 difiere esa infraestructura).
- **Store local.** `mlruns/` no se versiona; es reproducible mientras existan reportes y modelos.

## 6. Verificacion

- `tests/test_tracking.py` (9 tests): creacion de runs por fase, jerarquia padre/hijo para configs, registro de modelo, tags `test_usado`, idempotencia, `--force`, y rechazo de fase desconocida. Los tests usan un store SQLite temporal y un modelo toy.
- Re-ejecucion real: primera pasada 10 runs creados (padres); tras corregir la lectura de `busqueda` desde `search.json`, 40 runs mas (configs); tercera pasada 0 creados, 50 omitidos.
- `run_tracking.py` **no importa `load_splits` ni lee `test`**; solo lee reportes ya generados.
