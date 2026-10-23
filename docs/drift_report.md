# Monitorizacion de drift (Fase Q)

Fecha: 2026-10-22. Alcance: detectar cambios en las features que consume el
modelo, sin ground truth y **sin cargar `test`**. Referencia construida solo con
`train`; `validation` se usa unicamente como comprobacion descriptiva.

## 1. Que se monitorea

Se monitorea la **matriz preprocesada que ve el modelo**, no la transaccion raw.
El pipeline de inferencia ya reconstruye preprocessing + feature engineering y
verifica el orden de las 791 features; el monitor recibe esa misma fila mediante
un callback opcional (`InferencePipeline.predict(..., observe=...)`). Asi el
drift medido es exactamente el que puede afectar a la prediccion y no depende de
como se serialice una categoria en el body HTTP.

La referencia incluye las **25 features de mayor ganancia** de LightGBM, con 10
bins por cuantiles (bins degenerados se colapsan). `TransactionDT` se excluye de
forma explicita: es un indice temporal monotonico y con un split temporal su PSI
siempre seria enorme, tapando el resto de las features. El drift temporal sigue
representado por features de comportamiento (`since_first_sec_card1`,
`gap_sec_card1`, agregados historicos, etc.).

Cada feature incluye un bucket final de **missing / no numerico**. Los bins y sus
proporciones se guardan en `artifacts/drift_reference.json` (gitignored) con
metadatos de procedencia: `source=train`, `split_sha256`, filas disponibles,
filas muestreadas, seed, tipo de importancia y `test_usado=false`.

## 2. Referencia generada

Ejecucion:

```text
uv run run_drift.py
```

- Fuente: `train`, muestra aleatoria determinista de 50.000 filas (`seed=42`) de
  434.176; no se usa `validation` ni `test` para construirla.
- Features: 25, por ganancia del modelo desplegado (T-E02, config 11).
- Bins: 10 por feature, con bucket de missing.
- SHA-256 del split:
  `162c566a169f740f536c38070343f167b67343536b8dd25be8f79acc8a5d7b5c`.
- Salida: `artifacts/drift_reference.json`; resumen reproducible en
  `reports/drift/summary.json` (gitignored).

Features monitorizadas:

```text
V258, C1, C14, card1, cnt_P_emaildomain, TransactionAmt, C13, card2, V294,
cnt_client_uid, cnt_addr1, C4, addr1, std_amt_card1_prev,
mean_amt_card1_prev, cnt_card1, amt_rel_card1, C8, D2, D15, cnt_card1_addr1,
nunique_addr1_card1, since_first_sec_card1, P_emaildomain, gap_sec_card1
```

## 3. PSI y estados

El monitor acumula observaciones en memoria y calcula **Population Stability
Index** contra las proporciones de referencia:

```text
PSI = suma((actual - esperado) * ln(actual / esperado))
```

Los buckets vacios en ambas distribuciones se ignoran; si un bucket solo existe
en una, se usa `epsilon=1e-6` para evitar `log(0)`. Umbrales operativos:

| Estado | PSI |
|---|---|
| `stable` | < 0,10 |
| `moderate` | 0,10 a < 0,25 |
| `significant` | >= 0,25 |
| `no_data` | sin observaciones |

La ventana es exponencial: al superar `max_observations` (10.000 por defecto) se
conserva la mitad de los conteos. No crece sin limite y el PSI refleja trafico
reciente. El estado es **por proceso**: se reinicia al reiniciar la API y no se
persiste; para series historicas hace falta un store externo.

## 4. Comprobacion descriptiva con validation

`run_drift.py` observa las 77.822 filas completas de `validation` con una ventana
suficientemente grande. Resultado: estado global `significant`, PSI maximo
**4,95893**. Las mayores diferencias son:

| Feature | PSI | Estado |
|---|---:|---|
| `since_first_sec_card1` | 4,95893 | significant |
| `cnt_P_emaildomain` | 3,06044 | significant |
| `cnt_addr1` | 0,88074 | significant |
| `cnt_client_uid` | 0,25492 | significant |
| `cnt_card1` | 0,23894 | moderate |
| `cnt_card1_addr1` | 0,19963 | moderate |
| `nunique_addr1_card1` | 0,11040 | moderate |

Esto es esperable: `validation` es posterior a `train` y las features historicas
acumulan mas eventos. **No se usa para seleccionar nada** ni invalida el modelo;
sirve para verificar que el monitor detecta cambios reales y para documentar que
la referencia de produccion deberia actualizarse con trafico reciente cuando el
equipo de operaciones lo considere. `test` no se carga en ningun momento.

## 5. API y metricas

Con `drift_reference` en `config/deployment.json`, la API carga el monitor al
arrancar. Si la ruta esta configurada y el archivo no existe, la app no arranca.

| Endpoint | Uso |
|---|---|
| `GET /drift` | Estado, PSI maximo y detalle por feature. `enabled=false` si no hay referencia. |
| `GET /metrics` | Ademas de las metricas de Fase N, expone gauges de drift. |

Gauges Prometheus actualizados en cada scrape de `/metrics`:

| Metrica | Etiquetas |
|---|---|
| `fraud_drift_psi` | `feature` |
| `fraud_drift_observations` | sin etiquetas |
| `fraud_drift_features_drifted` | sin etiquetas (PSI >= 0,25) |
| `fraud_drift_max_psi` | sin etiquetas |

`/drift` documenta su contrato en Swagger y devuelve JSON. El monitor no bloquea
predicciones: si el callback falla, se registra un warning y la prediccion
continua.

## 6. Verificacion

- Tests nuevos de PSI, bins, missing, features constantes, serializacion JSON,
  deteccion de shift, ventana y estados.
- Tests HTTP: `/drift` deshabilitado sin referencia y habilitado con monitor;
  una prediccion incrementa las observaciones y `/metrics` publica los gauges.
- `run_drift.py` ejecutado sobre `train` + `validation`; `test_usado=false` en el
  resumen. Dos ejecuciones producen el mismo SHA-256 en la referencia
  (`47208937...4bf49`) y en el resumen (`e5cd438c...61cdd`).
- La API real arranca con `artifacts/drift_reference.json` y `/drift` responde
  `enabled=true`, `status=no_data`, 25 features.

## 7. Limites

- No mide performance del modelo: no hay ground truth en linea. Un PSI alto es
  una senal de entrada, no una prueba de degradacion.
- La referencia es una foto de `train`; no se recalibra automaticamente. Un
  proceso operativo debe decidir cuando regenerarla con trafico reciente.
- El estado vive en memoria y por proceso; con varios workers cada uno mantiene
  su propia ventana. Para agregacion hace falta un store compartido.
- Una feature constante se representa con una frontera sintetica alrededor del
  valor observado; cambios a otro valor se detectan, pero la granularidad es
  limitada.
