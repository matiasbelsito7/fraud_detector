# Operaciones: Docker, CI y monitoring (Fase N)

Fecha: 2026-10-19. Alcance: T-N01 (Docker), T-N02 (tests automatizados), T-N03 (CI/CD) y T-N04 (monitoring). `test` no se carga.

## 1. Docker

Archivos:

- `Dockerfile`: imagen `python:3.14-slim` con `uv 0.12.5`, dependencias sincronizadas desde `uv.lock`, `libgomp1` (necesario para LightGBM), usuario no root `uid 10001`, healthcheck y `run_api.py` como comando.
- `docker-compose.yml`: publica `8000`, monta `./artifacts` read-only en `/app/artifacts` y fija `FRAUD_DEPLOYMENT_CONFIG=/app/config/deployment.json`.
- `.dockerignore`: excluye datos, reportes, artefactos, secretos, tests, notebooks y entornos locales.

Los artefactos **no se hornean en la imagen** (estan gitignored y superan el tamano razonable de un contexto): se montan en runtime. Si faltan o no coinciden con los sha256 de `deployment.json`, la app no arranca.

```text
docker build -t fraud-detector-api:local .
docker run --rm -p 8000:8000 \
  -v "${PWD}/artifacts:/app/artifacts:ro" \
  fraud-detector-api:local
```

Con Compose:

```text
docker compose up --build
```

El healthcheck consulta `GET /health` cada 30 s. La imagen expone `/docs`, `/health`, `/predict` y `/metrics`.

**Verificacion local:** el primer build completo y el contenedor arrancaron, pero fallaron al cargar LightGBM por falta de `libgomp1`; se agrego la dependencia al `Dockerfile`. El segundo build recompilo todas las capas y llego hasta la exportacion, donde Docker Desktop se detuvo porque el host tenia **284 MB libres**. El smoke final del contenedor no se pudo repetir en este host. La construccion quedo validada en CI: run `39338449866`, job `docker` en 3m57s sobre `ubuntu-latest`.

## 2. Tests automatizados

- Suite completa: **159 tests** (`tests/`, actualizado en Fase Q), incluidos API, inferencia, preprocessing, features, split, modelado, tuning, evaluacion, explainability, tracking, categoricas, robustez y drift.
- Tests HTTP en `tests/test_api.py`: health, prediccion con/sin historial, historial futuro, validacion de body, threshold, identity sin ID, metricas y drift.
- Tests de drift en `tests/test_drift.py`: PSI, bins, missing, constantes, serializacion y ventana.
- Los tests no requieren artefactos reales: `tests/conftest.py` construye un pipeline de juguete determinista.

## 3. CI/CD

`.github/workflows/ci.yml` se ejecuta en push a `main`, pull requests y manualmente:

1. **quality**: `uv sync --locked`; ruff, black, mypy y `pytest`.
2. **docker**: build de la imagen con BuildKit y cache de GitHub Actions; no publica.

No hay despliegue automatico: no existe target, registry ni credenciales declaradas. Publicar la imagen o desplegar queda como decision de infraestructura. El job `docker` valido que el `Dockerfile` construye en Linux (run `39338449866`, commit `5cfeb43`), que es la verificacion que no pudo completarse localmente por espacio en disco.

## 4. Monitoring

La API expone:

| Endpoint | Uso |
|---|---|
| `GET /health` | Liveness/readiness y datos del modelo (variante, config, umbral). |
| `GET /drift` | PSI por feature contra la referencia de `train`; `enabled=false` si no hay referencia. |
| `GET /metrics` | Metricas Prometheus en texto. |

Metricas:

| Metrica | Tipo | Etiquetas |
|---|---|---|
| `fraud_api_requests_total` | counter | `method`, `route`, `status` |
| `fraud_api_request_duration_seconds` | histogram | `method`, `route` |
| `fraud_api_predictions_total` | counter | `label` (`fraude`/`legitima`) |
| `fraud_drift_psi` | gauge | `feature` |
| `fraud_drift_observations` | gauge | - |
| `fraud_drift_features_drifted` | gauge | - |
| `fraud_drift_max_psi` | gauge | - |

### Drift de features (Fase Q)

La referencia se construye solo con `train`:

```text
uv run run_drift.py
```

El script escribe `artifacts/drift_reference.json` (gitignored) con las 25
features de mayor ganancia, 10 bins por feature y bucket de missing. `test` no se
carga; `validation` solo se contrasta de forma descriptiva. `config/deployment.json`
declara `drift_reference`; si la ruta esta configurada y el archivo falta, la API
no arranca. El monitor observa la fila preprocesada que consume el modelo, no la
transaccion raw. Detalle, resultados de `validation` y limites en
`docs/drift_report.md`.

Ejemplo de scrape:

```yaml
scrape_configs:
  - job_name: fraud-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

`/metrics` no aparece en Swagger. Las metricas son por proceso: con varios workers hay que usar el modo multiproceso de `prometheus_client` o agregar por instancia.

**Fuera de alcance:** performance del modelo y metricas de negocio requieren
ground truth diferido y un store de predicciones. El drift de features de Fase Q
es una senal de entrada, no una medida de degradacion.

## 5. Limites

- La imagen instala todas las dependencias del proyecto (incluye MLflow y Jupyter), por lo que es grande; separar grupos de dependencias para la API queda como optimizacion.
- Los artefactos deben estar presentes en el host y coincidir con `deployment.json`.
- El contenedor no incluye TLS, autenticacion ni rate limiting; eso corresponde al gateway.
- CI no publica imagenes ni despliega.
- El smoke de Docker local quedo pendiente por falta de espacio en disco; CI es la verificacion vigente.
- MLflow sigue siendo un store local; no se levanta un servidor/registry (mencionado en `specs.md §13` como evolucion posterior).
