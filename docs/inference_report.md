# Pipeline de inferencia (Fase L)

Fecha: 2026-10-19. Modulos: `src/inference/pipeline.py`; configuracion: `config/deployment.json`; demostracion: `run_inference.py`. Evidencia (gitignored): `reports/inference/example.json`.

## 1. Decision de despliegue

Se despliega el modelo **T-E02** (`artifacts/model_final.joblib`, config 11, 439 rondas, 791 features) con el umbral **0,008480**, fijado unicamente en `validation` (`run_threshold.py`); `test` no se uso para elegirlo.

Motivo: en `validation` tiene mejor PR-AUC que el modelo de Fase H (0,6079 frente a 0,5814) y alcanza el mismo recall objetivo (0,8002) con **11,30 % de alertas frente a 12,87 %**. La mejora de las 29 features de cliente **sigue sin certificarse** (`docs/client_features_report.md`), pero la eleccion operativa se apoya en `validation`, no en `test`. El modelo de Fase H conserva las unicas metricas de `test` (ROC-AUC 0,9025, recall 0,7707) y es el que se reporta como estimacion de desempeno en produccion, porque `test` no se puede volver a leer para validar T-E02.

`test` no se uso en esta fase.

## 2. Contrato de entrada

`InferencePipeline.predict(transaction, history=None, identity=None, history_identity=None, threshold=None)`:

| Entrada | Requerida | Contenido |
|---|---|---|
| `transaction` | si | Una transaccion raw (mapeo, `Series` o `DataFrame` de una fila). Debe traer `TransactionDT`, `TransactionAmt` y `card1` no nulos. |
| `history` | no | Transacciones anteriores del cliente. Se rechaza cualquier fila con `TransactionDT` posterior a la transaccion (leakage temporal). |
| `identity` | no | Fila de `train_identity` de la transaccion nueva; requiere `TransactionID`. |
| `history_identity` | no | Identidad de las filas del historial (solo afecta a features de dispositivo historicas). |
| `threshold` | no | Anula el umbral de `deployment.json` para esa llamada. |

Las columnas que falten se rellenan de forma **explicita y uniforme**: numericas con `NaN` (el `SimpleImputer` del entrenamiento aplica la mediana) y categoricas con el centinela `__inference_missing__`, que `OrdinalEncoder` codifica como `-1`, igual que `encoded_missing_value` en entrenamiento. No se imputa nada con estadisticas calculadas en inferencia.

## 3. Transformaciones

1. Se reconstruye el esquema de las 390 columnas de entrada desde `preprocessor.feature_names_in_`.
2. Se concatena `history` + `transaction`, se ordena por `TransactionDT` (estable) y se marca la fila a puntuar.
3. `preprocessor.joblib` transforma esa fila (imputacion, `log1p` + `RobustScaler` para el importe, encoding ordinal y passthrough de `TransactionDT`).
4. `compute_features` calcula tiempo, frecuencia, historial de tarjeta, entidades y las 29 features de cliente sobre el historial + la fila nueva.
5. Se concatena preprocessing + features y se **verifica que el conjunto y el orden de columnas coincidan exactamente con `feature_name_` del modelo**; si falta o sobra una, aborta.
6. Se castea a `float32` (el dtype del parquet de entrenamiento) y se aplica `predict_proba`.
7. Se compara la probabilidad con el umbral configurado y se devuelve `Prediction` (probabilidad, clase, etiqueta, umbral, variante y filas de historial).

## 4. Configuracion desplegable

`config/deployment.json` es autocontenido (no depende de `reports/`, gitignored): variante, rutas de modelo y preprocessor, **sha256 de ambos artefactos**, umbral, `config_id`, rondas, `split_sha256` y `target_recall`. `Deployment.load` aborta si un artefacto no coincide con su digest, de modo que un modelo reentrenado no se sirve con un umbral viejo por accidente.

## 5. Verificacion end-to-end

`uv run run_inference.py`:

- carga `train_transaction.csv` e `train_identity.csv`;
- elige la primera transaccion de `validation` (`TransactionID=3421176`, `TransactionDT=10972801`);
- le pasa como historial las **434.176 filas anteriores**, sin ninguna futura;
- predice con T-E02: probabilidad `0,000324`, clase `legitima`;
- compara contra la probabilidad que el pipeline por lotes asigno a esa misma fila en `split.parquet`: **diferencia 0,00e+00**.

La coincidencia exacta valida que preprocessing, feature engineering, orden de columnas y dtype son los mismos en inferencia y entrenamiento (`specs.md` §11, §15). La fila es de `validation`; `test` no se carga en esta prueba.

## 6. Limites y riesgos

- **El historial es responsabilidad del entorno de produccion.** El pipeline es sin estado: recibe el historial como argumento. Sin historial, las features historicas usan centinelas (`cnt_*=0`, medias `0.0`, `gap=-1`) y la prediccion pierde senal, aunque no falla. Un despliegue real necesita un store de transacciones por cliente/tarjeta.
- **Coste de `compute_features`.** Recalcula agregados sobre todo el historial recibido. Para una sola transaccion es asumible; para alto volumen convendria precalcular agregados incrementales.
- **Artefactos no versionados.** `artifacts/*.joblib` esta gitignored; en un clon limpio hay que regenerarlos (`run_preprocessing.py`, `run_tuning.py`) antes de inferir.
- **Identidad opcional.** Sin fila de identity se pierden `has_identity`/`is_mobile`; el modelo fue entrenado con esos valores centinela cuando faltaban, pero la prediccion puede degradarse.
- **Categoricas ausentes.** El centinela produce `-1`, igual que el missing de entrenamiento; no se distingue "columna no enviada" de "valor nulo", que es el comportamiento mas conservador.

## 7. Tests

`tests/test_inference.py` (7 tests): consistencia fila unica vs lote, rechazo de historial futuro, relleno de columnas faltantes, umbral configurable, validacion de digest, validacion de umbral y comprobacion de que el codigo de inferencia no importa `load_splits` ni lee `test`.
