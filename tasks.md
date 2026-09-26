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

> Completada (2026-09-25): features en `src/features/`, documentación en `docs/features_report.md`, tests sin leakage.

### T-E01 Implementar componente de feature engineering
- **Objetivo:** generar features fundamentadas sobre tiempo, frecuencia, historial, importes, entidades y relaciones.
- **Descripción:** implementar el componente separado (`specs §5`), documentando cada feature relevante (qué representa, cómo se calcula, qué información utiliza, disponibilidad en inferencia).
- **Dependencias:** T-C01.
- **Archivos/componentes probables:** módulo de feature engineering, documentación de features, tests.
- **Verificación (specs §15 — features consistentes, sin leakage):** las features se generan consistentemente en entrenamiento e inferencia, y se comprueba que ninguna usa información futura.

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

## Fase J — Explainability

### T-J01 Analizar explicabilidad global e individual
- **Objetivo:** entender las predicciones del modelo.
- **Descripción:** feature importance global y SHAP (global e individual cuando sea posible); artefactos asociados al experimento (`specs §9`).
- **Dependencias:** T-I01.
- **Archivos/componentes probables:** reporte de importancia, análisis SHAP, artefactos exportados.
- **Verificación:** existen artefactos de explainability reproducibles y asociados al modelo seleccionado.

## Fase K — Experiment tracking (MLflow)

Transversal a las Fases G–J.

### T-K01 Registrar experimentos en MLflow
- **Objetivo:** trazabilidad y comparación de experimentos.
- **Descripción:** registrar experimentos, parámetros, métricas, modelos y artefactos relevantes (`specs §10`).
- **Dependencias:** T-G01 (y se alimenta de G–J).
- **Archivos/componentes probables:** integración MLflow en entrenamiento/evaluación, configuración del tracking.
- **Verificación (specs §15 — experimentos registrados):** cada experimento relevante queda registrado con código, datos, configuración y resultados asociados, reproducible y comparable.

## Fase L — Pipeline de inferencia

### T-L01 Implementar inferencia consistente con entrenamiento
- **Objetivo:** que una nueva transacción atraviese todo el pipeline.
- **Descripción:** recibir una transacción, aplicar exactamente las transformaciones del entrenamiento, generar probabilidad, aplicar threshold configurable y devolver predicción interpretable (`specs §11`).
- **Dependencias:** T-D01, T-E01, T-I01.
- **Archivos/componentes probables:** módulo de inferencia, pipeline compartido de transformaciones, tests de consistencia.
- **Verificación (specs §15 — pipeline completo, inferencia consistente):** una transacción atraviesa el pipeline de punta a punta y la inferencia utiliza el mismo procesamiento que el entrenamiento.

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

### T-O01 Revisión cruzada contra criterios de aceptación
- **Objetivo:** confirmar que el proyecto cumple `specs §15`.
- **Descripción:** verificar etapa por etapa los criterios de aceptación y completar la documentación pendiente del repositorio.
- **Dependencias:** todas las fases obligatorias.
- **Archivos/componentes probables:** actualización de README/docs, checklist de aceptación.
- **Verificación:** todos los criterios de `specs §15` se cumplen o quedan documentados los pendientes y riesgos conocidos.