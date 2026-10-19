# tasks.md

Plan de implementación del proyecto de detección de fraude (IEEE-CIS Fraud Detection). Convierte las especificaciones de `specs.md` en tareas ordenadas e incrementales: indica **qué hacer, en qué orden y cómo verificar** que cada etapa está terminada.

Este documento no redefine principios (`constitution.md`), reglas de trabajo (`agents.md`) ni requisitos (`specs.md`).

## Cómo usar este documento

- Seguir las fases en orden; no avanzar a la siguiente sin completar la verificación de la anterior.
- Cada tarea verifica su fin de etapa mediante los criterios indicados, enlazados con `specs.md §15` cuando corresponda.
- Las fases marcadas como **diferidas** (M y N) están fuera del alcance inicial y solo se ejecutan cuando las etapas previas estén estables.

## Fase A — Preparación del repositorio

### T-A01 Configurar estructura base del repositorio
- **Objetivo:** definir una organización inicial que soporte desarrollo incremental.
- **Descripción:** crear la estructura de carpetas y archivos de configuración mínima (sin fijar arbitrariamente decisiones no tomadas en `specs.md §14`).
- **Dependencias:** ninguna.
- **Archivos/componentes probables:** `.gitignore`, `pyproject.toml` (declaración de dependencias), README breve, carpeta de datos (sin versionar), carpetas de código y config.
- **Verificación:** se puede clonar/abrir el repo y ejecutar un comando de referencia sin errores.

### T-A02 Configurar entorno de Python y dependencias
- **Objetivo:** disponer de un entorno reproducible para todo el desarrollo.
- **Descripción:** definir versiones de Python y de librerías core (pandas, numpy, scikit-learn, y las que se incorporen según fase).
- **Dependencias:** T-A01.
- **Archivos/componentes probables:** dependencias declaradas en `pyproject.toml`, entorno virtual gestionado con `uv`, lock de versiones (`uv.lock`).
- **Verificación:** un script trivial que importe las librerías principales se ejecuta sin errores.

## Fase B — Adquisición y validación de datos

### T-B01 Incorporar y validar los datos
- **Objetivo:** disponer de `train_transaction.csv` y `train_identity.csv` verificados.
- **Descripción:** cargar ambos archivos (sin versionarlos), confirmar la estructura real y registrar los hallazgos. Confirmar al cargar: número de filas (~590.000 de referencia), tasa de fraude (~3,5 %), tipos de columnas, rango de `TransactionDT`, cobertura del join por `TransactionID`.
- **Dependencias:** T-A02.
- **Archivos/componentes probables:** carpeta de datos, script de carga/inspección, reporte de validación.
- **Verificación (specs §15 — datos validables):** ambos archivos se leen correctamente, el join por `TransactionID` es consistente y la calidad (tipos, missings, distribución de target) queda documentada.
- **Resultado:** `reports/validation_report.json` (gitignored): 590.540 x 394 transacciones y 144.233 x 41 identity; `TransactionID` int64, único y sin nulos en ambos; join 24,42 % y 0 identity huérfanas; target 3,50 %; rango temporal y missing documentados. Síntesis en `docs/eda_report.md`.

## Fase C — Data understanding y EDA

> Completada (2026-09-25): EDA ejecutado; conclusiones en `docs/eda_report.md`, resultados en `reports/eda/`.

### T-C01 Ejecutar y documentar el EDA
- **Objetivo:** conocer los datos y fundamentar las decisiones de preprocessing y feature engineering.
- **Descripción:** analizar tipos, missing values, cardinalidad, distribuciones, distribución de fraude, comportamiento temporal, relaciones relevantes y anomalías, conforme `specs §3`.
- **Dependencias:** T-B01.
- **Archivos/componentes probables:** notebook o script de EDA, reporte/notebook compartido con conclusiones escritas.
- **Verificación:** el EDA produce conclusiones escritas que fundamentan las decisiones de las Fases D y E; no se fijan transformaciones sin ese fundamento.

## Fase D — Preprocessing

> Completada (2026-09-25): pipeline reproducible en `src/preprocessing/`, tests en `tests/`, decisiones en `docs/preprocessing_report.md`.

### T-D01 Construir pipeline reproducible de preprocessing
- **Objetivo:** transformar los datos de forma reproducible y reusable en inferencia.
- **Descripción:** implementar el pipeline según lo que determine el EDA (`specs §4`): missing values, categóricas, numéricas, encoding, scaling, eliminación/conservación justificada de variables.
- **Dependencias:** T-C01.
- **Archivos/componentes probables:** módulo de preprocessing, tests asociados, artefacto serializado del pipeline.
- **Verificación (specs §15 — preprocessing reproducible):** el pipeline se ejecuta de forma idéntica al menos dos veces sobre los mismos datos y puede reutilizarse en inferencia.

## Fase E — Feature engineering

> Completada (2026-09-25): features en `src/features/`, documentación en `docs/features_report.md`, tests sin leakage. Ampliada el 2026-09-28 con T-E02 (identidad de cliente, 29 features), implementada y medida pero **no certificada como mejora**: ver T-E02 más abajo y `docs/client_features_report.md`.

### T-E01 Implementar componente de feature engineering
- **Objetivo:** generar features fundamentadas sobre tiempo, frecuencia, historial, importes, entidades y relaciones.
- **Descripción:** implementar el componente separado (`specs §5`), documentando cada feature relevante (qué representa, cómo se calcula, qué información utiliza, disponibilidad en inferencia).
- **Dependencias:** T-C01.
- **Archivos/componentes probables:** módulo de feature engineering, documentación de features, tests.
- **Verificación (specs §15 — features consistentes, sin leakage):** las features se generan consistentemente en entrenamiento e inferencia, y se comprueba que ninguna usa información futura.

### T-E02 Agregar identidad de cliente como clave de feature engineering
- **Objetivo:** capturar el comportamiento a nivel de cliente (tarjeta + dirección), que es donde la etiqueta está definida realmente, en lugar de tratar cada transacción como independiente.
- **Contexto:** el objetivo de negocio de recall ≥ 0,80 no se cumplió en `test` (recall 0,7707, IC 95 % `[0,7551, 0,7864]`) y la ROC-AUC quedó en 0,9025 frente a ~0,946 del ganador de la competencia IEEE-CIS. La causa identificada es que no tenemos features de identidad de cliente: la referencia que alcanzó esa cifra usó un UID derivado de tarjeta/dirección/día de alta. Análisis completo y datos de soporte en `docs/client_features_report.md`.
- **Descripción:** agregar la media previa de C1–C14 por dos claves de agrupación. **Primaria:** `card1 + addr1`. **Refinada:** `card1 + addr1 + D1n`, donde `D1n = TransactionDay - D1`. La media previa es acumulativa y se calcula vectorizada con `groupby(keys).cumsum() - x` sobre filas **previas** del grupo. Sin historial → centinela `0.0`, nunca `NaN`, siguiendo la convención de `mean_amt_card1_prev`. **29 features nuevas** (20 → 49 propias; parquet de 764 a 793 columnas). Módulo `src/features/client.py`, nombres de columnas crudas compartidos en `src/features/columns.py`.
- **`D1n` es clave interna, no feature:** su información ya está en `D1` (la edad de la tarjeta *es* `D1`), y emitirlo reintroduciría el identificador de cliente que la referencia explícitamente rechaza.
- **Decisiones tomadas (confirmadas por el usuario):**
  - Clave primaria `card1+addr1`, no el UID `card1+addr1+D1n` de la referencia, porque aquí el UID completo solo cubriría 34,4 % de las filas de test.
  - Solo columnas C. V (339 columnas) explotaría la dimensionalidad sin evidencia que lo justifique. **M1–M9 además se excluyeron al implementar** (el diseño inicial decía C+M): en crudo son categóricas, las codifica el preprocessor, M1 es `T` en el 99,99 % de sus no nulos y entre el 46 % y el 59 % de las M son nulas, luego su "media previa" no está definida ni informa. Fijado con `test_columnas_m_no_se_agregan`.
  - Solo media, sin desviación estándar, por simplicidad (`constitution.md` §7).
- **Dependencias:** T-E01.
- **Archivos/componentes probables:** `src/features/client.py` (nuevo), `src/features/columns.py` (nuevo), `src/features/features.py`, `run_feature_engineering.py`, `docs/features_report.md`, `docs/client_features_report.md`, tests en `tests/test_features.py`.
- **Verificación (specs §15 — features consistentes, sin leakage):** seis tests nuevos. El más crítico es `test_media_vectorizada_coincide_con_expanding`, que ata el atajo `cumsum` a `previous_expanding` sobre un caso pequeño, para que la optimización no pueda divergir de la definición. Los otros: `test_clave_refinada_separa_mismo_cliente`, `test_agregaciones_de_cliente_sin_info_futura`, `test_cliente_sin_historial_usa_centinela`, `test_degrada_sin_columnas_c_m_d1`, `test_columnas_m_no_se_agregan` y extensión de `test_inferencia_fila_unica_consistente`.
- **Estado: IMPLEMENTADA Y MEDIDA. No certificada como mejora** (2026-09-28). `run_tuning.py` y `run_threshold.py` re-ejecutados; `run_evaluation.py` no. Resultado y veredicto en `docs/client_features_report.md`.

#### Regla de certificación de T-E02 (crítica, no negociable)

Añadir features es **selección de features**, y `constitution.md` §5 exige que el conjunto de test se mantenga independiente de ella. `test` ya fue consumido en la Fase I (T-I01).

| Paso | ¿Se ejecuta? | Motivo |
|---|---|---|
| `run_feature_engineering.py` | Sí | Reconstruye el parquet con las features nuevas |
| `run_tuning.py` (T-H01) | Sí | Selecciona hiperparámetros sobre el nuevo set; objetivo PR-AUC OOF > 0,633573 |
| `run_threshold.py` | Sí | Fija el umbral nuevo en `validation` + corroboración OOF |
| `run_evaluation.py` | **NO** | `test` no se vuelve a leer bajo ninguna circunstancia |

Consecuencia que debe quedar escrita, no implícita: **la cifra de `test` (ROC-AUC 0,9025 / PR-AUC 0,5282 / recall 0,7707) corresponde al modelo baseline sin features de cliente.** El modelo con T-E02 se certifica únicamente con train out-of-fold y `validation`, y hay que reportarlo así en `docs/evaluation_report.md` y en el `README.md`.

La comparación del nuevo set contra el baseline se hace **por fold, no solo por media**: si la mejora de las features de cliente no es consistente en los 3 pliegues walk-forward, es ruido.

#### Resultado de la certificación (2026-09-28)

| Comparación (PR-AUC OOF) | T-E02 | Fase H | Δ media | Δ por pliegue |
|---|---:|---:|---:|---|
| Ganadora | 0,640227 (config 11, 439 rondas) | 0,633573 (config 18, 219 rondas) | +0,006654 | **[+0,0104, +0,0110, −0,0014]** |
| Ancla (mismos hiperparámetros) | 0,630059 | 0,631655 | −0,001596 | **[−0,0089, +0,0086, −0,0044]** |

La meta (OOF > 0,633573) se cumple, pero **la mejora no es consistente en los 3 pliegues y la comparación aislada es negativa**: la diferencia entre las dos búsquedas se explica por la configuración elegida, no por las 29 features. Por la regla de este documento, **T-E02 no se certifica como mejora**. Descartada por medición la hipótesis de cobertura (claves con 94–95,7 % y 40,6–44,8 % de cobertura por pliegue, estable). Las features sí se usan (13,3 % de la ganancia; `cnt_client_uid` es la #10 del modelo), y el punto operativo de `validation` mejora (mismo recall con 11,30 % de alertas frente a 12,87 %).

Evidencia completa en `docs/client_features_report.md`; segunda ejecución en `docs/tuning_report.md` §10.

#### Pendiente de decisión

1. **Qué modelo se despliega:** resuelto en Fase L: se despliega `artifacts/model_final.joblib` (T-E02) con umbral `0,008480`, por su mejor punto operativo en `validation` (mismo recall con 11,30 % de alertas frente a 12,87 %). No es una mejora certificada: las métricas de `test` siguen correspondiendo a Fase H. `artifacts/model_final_fase_h.joblib` se conserva como referencia. Detalle en `docs/inference_report.md`.
2. **Guarda técnica de `test`:** al re-ejecutarse `run_threshold.py`, `threshold.json` se regeneró con el digest vigente y **la precondición de digest de `run_evaluation.py` dejó de proteger**. Resuelto (2026-09-27): se añadió el candado versionado `config/test_consumed.json`, que registra cada consumo de `test` con independencia de los datos; `run_evaluation.py` aborta si ya figura consumido, salvo `--force` explícito que deja traza. El consumo de la Fase I quedó registrado como primer asiento.
3. **Numeración:** T-E02 queda dentro de la Fase E, como está, para no romper la secuencia y dejar explícito que el consumo de test no se repite.
4. **Hallazgo adyacente, fuera de alcance:** `card1`, `addr1`, `card2`, `card3`, `addr2`, `D1`, `P_emaildomain` y `ProductCD` están guardadas como **float**, así que LightGBM corta por umbrales numéricos sobre identificadores que no tienen orden. Nunca se trataron como categóricas. Es un cambio de una línea por columna y probablemente mueva el AUC tanto como T-E02, pero es otra decisión y queda aparte.

## Fase F — División de datos

> Completada (2026-09-25): split temporal `train` 18 sem / `validation` 4 / `test` 4, determinista y aislado; validación en `tests/test_split.py`, reporte en `docs/split_report.md`.

### T-F01 Definir y validar el split temporal
- **Objetivo:** generar train/validation/test adecuados a datos transaccionales y temporales.
- **Descripción:** estrategia que evite información futura en el entrenamiento; test aislado y reproducible (`specs §6`).
- **Dependencias:** T-D01, T-E01.
- **Archivos/componentes probables:** módulo de split, registro de semilla/criterio temporal.
- **Verificación:** se puede regenerar la misma división de forma reproducible y el test no participa de decisiones de features ni tuning.

## Fase G — Baselines y modelado

> Completada (2026-09-25): escalera de modelos en `src/modeling/`, ejecución `run_training.py`, 19 tests, decisiones y resultados en `docs/modeling_report.md`. Comparación en `reports/modeling/comparison.{csv,json}`; dos ejecuciones completas dieron métricas idénticas. `test` no se carga. Pliegues walk-forward internos de `train` implementados (uso en Fase H).

### T-G01 Entrenar y comparar modelos progresivamente
- **Objetivo:** obtener baseline y modelos de referencia bajo condiciones equivalentes.
- **Descripción:** baseline simple → modelos clásicos → modelos de boosting (`specs §7`), comparables en la misma división/features/evaluación.
- **Dependencias:** T-F01.
- **Archivos/componentes probables:** módulo de entrenamiento, configuración de experimentos, script de comparación.
- **Verificación (specs §15 — entrenamiento reproducible):** los modelos pueden entrenarse y compararse con resultados reproducibles.

## Fase H — Tuning de hiperparámetros

> Completada (2026-09-26): espacio y búsqueda en `src/modeling/{space,tuning}.py`, ejecución `run_tuning.py`, 21 tests, búsqueda de 20 configs × 3 pliegues en `reports/tuning/search.{csv,json}` y modelo final en `artifacts/model_final.joblib`; selección y resultados en `docs/tuning_report.md`. `test` no se carga; `validation` se usó una sola vez, como confirmación. Resultado negativo en ganancia: la config ganadora supera al ancla de Fase G en +0,0019 PR-AUC, muy por debajo del ruido entre pliegues (±0,05).

### T-H01 Ajustar hiperparámetros y seleccionar modelo
- **Objetivo:** mejorar el modelo con base en criterios y sin contaminar el test.
- **Descripción:** tuning sobre validation cuando las etapas anteriores lo justifiquen; selección final con las mismas condiciones de evaluación (`specs §7`).
- **Dependencias:** T-G01.
- **Archivos/componentes probables:** configuración de tuning, registros de config y resultados.
- **Verificación (specs §15 — test aislado):** el test no se usa en el tuning; la selección queda justificada y reproducible.
- **Nota de ejecución:** la búsqueda se hizo sobre los pliegues walk-forward internos de `train` (no sobre `validation`), siguiendo `constitution.md §5`; `validation` quedó como confirmación única. Desviación documentada en `docs/tuning_report.md` §2.

## Fase I — Evaluación final

### T-I01 Evaluar el modelo final sobre test
- **Objetivo:** medir el desempeño final con el conjunto mínimo de métricas.
- **Descripción:** ROC-AUC, PR-AUC, recall en threshold operativo y matriz de confusión; análisis de thresholds y elección justificada del operativo (secundarias: precision, F1, ROC-AUC) (`specs §8`).
- **Dependencias:** T-H01.
- **Archivos/componentes probables:** script/notebook de evaluación, reporte de métricas, curvas.
- **Verificación (specs §15 — métricas calculables):** las métricas mínimas se calculan sin errores sobre el test aislado.
- **Estado: COMPLETADA.** Umbral operativo `0,029329` fijado en `validation` antes de leer `test`. Resultado en `test`: ROC-AUC 0,9025, PR-AUC 0,5282, recall 0,7707 [0,7551, 0,7864], precision 0,1708, 15,94 % de alertas. **El objetivo de recall ≥ 0,80 no se cumple**: hay deriva temporal (recall 0,8161 → 0,8002 → 0,7707 en train OOF / validation / test). Reporte en `docs/evaluation_report.md`; resultados en `reports/evaluation/`. `test` queda consumido y no debe reutilizarse para decidir.

## Fase J — Explainability

> Completada (2026-10-18): importancia global (gain/split) y SHAP (TreeSHAP nativo de LightGBM) en `src/explainability/`, ejecución `run_explainability.py`, 14 tests y reporte en `docs/explainability_report.md`; artefactos en `reports/explainability/` (gitignored). `test` no se carga; la explicabilidad es descriptiva y no participa de ninguna selección.

### T-J01 Analizar explicabilidad global e individual
- **Objetivo:** entender las predicciones del modelo.
- **Descripción:** feature importance global y SHAP (global e individual cuando sea posible); artefactos asociados al experimento (`specs §9`).
- **Dependencias:** T-I01.
- **Archivos/componentes probables:** reporte de importancia, análisis SHAP, artefactos exportados.
- **Verificación:** existen artefactos de explainability reproducibles y asociados al modelo seleccionado.
- **Resultado:** modelo explicado `artifacts/model_final.joblib` (config 11, T-E02, 439 rondas, 791 features) sobre una muestra determinista de 5.000 filas de `validation` (`seed=42`). Aditividad SHAP verificada (error máximo 2,25 × 10⁻⁸) y dos ejecuciones con SHA-256 idéntico. `V258` domina la ganancia (10,25 %); por `|SHAP|` el orden lo encabezan `C13`, `since_first_sec_card1` y `TransactionAmt`. Las 29 features de cliente aportan 13,28 % de ganancia y 15,48 % de `|SHAP|`, aunque su mejora sigue sin certificarse. 333 de los 355 indicadores de missing nunca se usan. Detalle y casos individuales (TP/FN/FP) en `docs/explainability_report.md`.

## Fase K — Experiment tracking (MLflow)

> Completada (2026-10-19): registro por **backfill** en `src/tracking/` + `run_tracking.py`; store SQLite en `mlruns/mlflow.db` (gitignored), 50 runs idempotentes, 9 tests y reporte en `docs/tracking_report.md`. No reentrena ni vuelve a leer `test`.

### T-K01 Registrar experimentos en MLflow
- **Objetivo:** trazabilidad y comparación de experimentos.
- **Descripción:** registrar experimentos, parámetros, métricas, modelos y artefactos relevantes (`specs §10`).
- **Dependencias:** T-G01 (y se alimenta de G–J).
- **Archivos/componentes probables:** integración MLflow en entrenamiento/evaluación, configuración del tracking.
- **Verificación (specs §15 — experimentos registrados):** cada experimento relevante queda registrado con código, datos, configuración y resultados asociados, reproducible y comparable.
- **Resultado:** Fases G–J registradas leyendo los JSON/CSV de `reports/` y `docs/`, sin reentrenar ni cargar `test`. Cada run lleva `git_commit` de la fase que lo produjo, `split_sha256`, `config_id`/rondas y tag `test_usado`; la Fase I es el único con `test_usado=true`. Los modelos seleccionados se registran en formato MLflow (flavor LightGBM). Idempotencia por `tags.tracking_key` (segunda ejecución: 0 creados, 50 omitidos).

## Fase L — Pipeline de inferencia

> Completada (2026-10-19): pipeline en `src/inference/pipeline.py`, configuración autocontenida en `config/deployment.json`, demostración en `run_inference.py`, 7 tests y reporte en `docs/inference_report.md`. Verificación real: una fila de `validation` con 434.176 transacciones de historial predice idéntico al pipeline por lotes (diferencia 0,00e+00). `test` no se carga.

### T-L01 Implementar inferencia consistente con entrenamiento
- **Objetivo:** que una nueva transacción atraviese todo el pipeline.
- **Descripción:** recibir una transacción, aplicar exactamente las transformaciones del entrenamiento, generar probabilidad, aplicar threshold configurable y devolver predicción interpretable (`specs §11`).
- **Dependencias:** T-D01, T-E01, T-I01.
- **Archivos/componentes probables:** módulo de inferencia, pipeline compartido de transformaciones, tests de consistencia.
- **Verificación (specs §15 — pipeline completo, inferencia consistente):** una transacción atraviesa el pipeline de punta a punta y la inferencia utiliza el mismo procesamiento que el entrenamiento.
- **Resultado:** `InferencePipeline` valida sha256 de modelo y preprocessor, reconstruye el esquema desde `feature_names_in_`, concatena historial + transacción, rechaza historial futuro, ejecuta preprocessing y feature engineering, verifica el orden exacto de las 791 features y aplica el umbral. El historial es opcional con degradación documentada (centinelas, sin error). La demostración real usa T-E02 y compara contra `split.parquet` sin diferencias.

## Fases diferidas

### Fase M — API (diferida)
- **T-M01 Exponer la inferencia como API (p. ej. FastAPI):** recibir datos de una transacción, ejecutar pipeline y modelo, devolver resultado y probabilidad (`specs §12`).

### Fase N — Docker, tests, CI/CD, monitoring (diferida)
- **T-N01 Docker:** contenerizar el servicio y componentes cuando aplique.
- **T-N02 Tests automatizados:** cobertura de los pipelines y componentes principales.
- **T-N03 CI/CD:** integración y despliegue automatizado.
- **T-N04 Monitoring:** seguimiento del comportamiento en producción.
- Incorporar cada componente solo cuando las etapas anteriores estén estables (`specs §13`).

## Fase O — Documentación y revisión final

> Completada (2026-10-19): revisión de los 9 criterios de `specs.md §15` con evidencia en `docs/acceptance_report.md`; README y `tasks.md` actualizados. 121 tests y pre-commit en verde. Sin cambios de modelo ni nueva lectura de `test`.

### T-O01 Revisión cruzada contra criterios de aceptación
- **Objetivo:** confirmar que el proyecto cumple `specs §15`.
- **Descripción:** verificar etapa por etapa los criterios de aceptación y completar la documentación pendiente del repositorio.
- **Dependencias:** todas las fases obligatorias.
- **Archivos/componentes probables:** actualización de README/docs, checklist de aceptación.
- **Verificación:** todos los criterios de `specs §15` se cumplen o quedan documentados los pendientes y riesgos conocidos.
- **Resultado:** los 9 criterios se cumplen con evidencia verificable. Se corrigió una etiqueta de cobertura en `docs/eda_report.md` (DeviceType 23,84 %; join con identity 24,42 %). Riesgos abiertos documentados: recall 0,7707 en `test`, T-E02 desplegado sin métrica de `test`, historial externo en producción, artefactos no versionados y fases M/N diferidas.
