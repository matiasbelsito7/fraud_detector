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

**Verificacion local:** el primer build completo y el contenedor arrancaron, pero fallaron al cargar LightGBM por falta de `libgomp1`; se agrego la dependencia al `Dockerfile`. El segundo build recompilo todas las capas y llego hasta la exportacion, donde Docker Desktop se detuvo porque el host tenia **284 MB libres**. El smoke final del contenedor no se pudo repetir en este host. La construccion queda cubierta por el job `docker` de CI en cada push/PR.

## 2. Tests automatizados

- Suite completa: **130 tests** (`tests/`), incluidos API, inferencia, preprocessing, features, split, modelado, tuning, evaluacion, explainability y tracking.
- Tests HTTP en `tests/test_api.py`: health, prediccion con/sin historial, historial futuro, validacion de body, threshold, identity sin ID y metricas.
- Los tests no requieren artefactos reales: `tests/conftest.py` construye un pipeline de juguete determinista.

## 3. CI/CD

`.github/workflows/ci.yml` se ejecuta en push a `main`, pull requests y manualmente:

1. **quality**: `uv sync --locked`; ruff, black, mypy y `pytest`.
2. **docker**: build de la imagen con BuildKit y cache de GitHub Actions; no publica.

No hay despliegue automatico: no existe target, registry ni credenciales declaradas. Publicar la imagen o desplegar queda como decision de infraestructura. El job `docker` valida que el `Dockerfile` construye en Linux, que es la verificacion que no pudo completarse localmente por espacio en disco.

## 4. Monitoring

La API expone:

| Endpoint | Uso |
|---|---|
| `GET /health` | Liveness/readiness y datos del modelo (variante, config, umbral). |
| `GET /metrics` | Metricas Prometheus en texto. |

Metricas:

| Metrica | Tipo | Etiquetas |
|---|---|---|
| `fraud_api_requests_total` | counter | `method`, `route`, `status` |
| `fraud_api_request_duration_seconds` | histogram | `method`, `route` |
| `fraud_api_predictions_total` | counter | `label` (`fraude`/`legitima`) |

Ejemplo de scrape:

```yaml
scrape_configs:
  - job_name: fraud-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

`/metrics` no aparece en Swagger. Las metricas son por proceso: con varios workers hay que usar el modo multiproceso de `prometheus_client` o agregar por instancia.

**Fuera de alcance:** drift de features, performance del modelo y metricas de negocio requieren ground truth diferido y un store de predicciones. No se implementan en esta fase.

## 5. Limites

- La imagen instala todas las dependencias del proyecto (incluye MLflow y Jupyter), por lo que es grande; separar grupos de dependencias para la API queda como optimizacion.
- Los artefactos deben estar presentes en el host y coincidir con `deployment.json`.
- El contenedor no incluye TLS, autenticacion ni rate limiting; eso corresponde al gateway.
- CI no publica imagenes ni despliega.
- El smoke de Docker local quedo pendiente por falta de espacio en disco; CI es la verificacion vigente.
- MLflow sigue siendo un store local; no se levanta un servidor/registry (mencionado en `specs.md §13` como evolucion posterior).
