# API de inferencia (Fase M)

Fecha: 2026-10-19. Implementacion: `src/api/` (`schemas.py`, `app.py`); ejecucion: `run_api.py`. Stack: FastAPI 0.149 y Uvicorn 0.54. `test` no se carga ni se usa.

## 1. Alcance

Expone el pipeline de la Fase L por HTTP:

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/health` | Variante, `config_id` y umbral del modelo cargado. |
| `GET` | `/metrics` | Metricas Prometheus (requests, latencia y predicciones por etiqueta). |
| `POST` | `/predict` | Puntua una transaccion con historial e identidad opcionales. |
| `GET` | `/docs` | Swagger UI autogenerado. |
| `GET` | `/openapi.json` | Esquema OpenAPI. |

No incluye autenticacion, TLS, rate limiting ni persistencia de predicciones: no estan en `specs.md` y se consideran responsabilidad del despliegue.

## 2. Contrato

### Request

```json
{
  "transaction": {
    "TransactionID": 3421176,
    "TransactionDT": 10972801,
    "TransactionAmt": 100.0,
    "card1": 1
  },
  "history": [
    {"TransactionID": 1, "TransactionDT": 10970000, "TransactionAmt": 50.0, "card1": 1}
  ],
  "identity": {"TransactionID": 3421176, "DeviceType": "mobile"},
  "history_identity": [],
  "threshold": 0.01
}
```

- `transaction` (obligatorio): una fila raw. Exige `TransactionDT`, `TransactionAmt` y `card1` no nulos. Las columnas ausentes se rellenan como en la Fase L.
- `history`: transacciones anteriores del cliente. Se rechaza cualquier `TransactionDT` posterior a la transaccion (leakage temporal).
- `identity`: fila de identity de la transaccion; si se envia, la transaccion debe traer `TransactionID`.
- `history_identity`: identity de las filas del historial.
- `threshold`: opcional, en `(0, 1)`; anula el umbral de `deployment.json`.
- Campos extra en el nivel superior: rechazados (`422`).

### Response

```json
{
  "probability": 0.000324,
  "is_fraud": false,
  "label": "legitima",
  "threshold": 0.008480293333969607,
  "variant": "te02",
  "config_id": 11,
  "n_history": 434176
}
```

## 3. Carga del modelo

- El modelo y el preprocessor se cargan **una sola vez** al arrancar (`lifespan`), desde `config/deployment.json` por defecto.
- La ruta se puede cambiar con `--config` o con la variable `FRAUD_DEPLOYMENT_CONFIG`.
- `InferencePipeline.load` verifica los sha256 de ambos artefactos: si no coinciden, el proceso no arranca.
- Para tests o integraciones se puede inyectar un pipeline ya construido: `create_app(pipeline=...)`.

## 4. Ejecucion

```text
uv run run_api.py
uv run run_api.py --host 0.0.0.0 --port 8001
uv run run_api.py --reload          # desarrollo
uv run uvicorn --factory src.api.app:create_app
```

Ejemplo:

```text
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"transaction\":{\"TransactionDT\":10972801,\"TransactionAmt\":100.0,\"card1\":1}}"
```

## 5. Errores

| Codigo | Caso |
|---|---|
| `400` | Historial con transacciones futuras; identity sin `TransactionID`; umbral invalido en el pipeline. |
| `422` | Body invalido: falta `transaction` o sus columnas requeridas, `threshold` fuera de rango, campos extra. |
| `500` | Error inesperado del modelo (no se enmascara). |

El arranque aborta si faltan los artefactos o si los digests no coinciden.

## 6. Verificacion

- `uv run pytest tests/test_api.py`: 9 tests con el pipeline de juguete de `tests/conftest.py` (health, request/response, historial, historial futuro, columnas requeridas, umbral, identity sin ID y metricas).
- Smoke real con `create_app()` y T-E02:
  - `GET /health` -> `200`, `variant=te02`, `config_id=11`, umbral `0,008480`;
  - `POST /predict` con una transaccion sin historial -> `200`, probabilidad `0,005315`, `label=legitima`, `n_history=0`.
- La prediccion real sin historial difiere de la version con historial (`0,000324`), como se espera por la degradacion documentada en `docs/inference_report.md`.

## 7. Limites

- El historial viaja en el body: para clientes con mucho historial el payload crece. En produccion conviene un store de historial y enviar solo la transaccion.
- El proceso es sincrono y CPU-bound; escalar con varios workers implica cargar una copia del modelo por worker.
- No hay autenticacion ni auditoria. Cualquier exposicion publica necesita gateway, TLS y controles de acceso.
- Un solo modelo por proceso: cambiar de variante exige reiniciar con otro `deployment.json`.
- La API no reemplaza `run_inference.py`; este sigue siendo la verificacion reproducible contra el pipeline por lotes.
