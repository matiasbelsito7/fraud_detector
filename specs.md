# specs.md

Especificaciones del sistema de detección de fraude. Define qué debe construirse, los requisitos concretos, el comportamiento esperado y los criterios de aceptación del proyecto.

Las reglas de trabajo de los agentes se definen en `agents.md`; los principios innegociables del proyecto, en `constitution.md`.

---

## 1. Objetivo del sistema

- Construir un sistema de Machine Learning capaz de estimar si una transacción es fraudulenta.
- Utilizar el dataset **IEEE-CIS Fraud Detection** como fuente de datos.
- Desarrollar el proyecto con orientación end-to-end de ML Engineering / MLOps, de forma progresiva e incremental.

## 2. Datos

### 2.1 Fuentes

El sistema debe trabajar con los archivos de entrenamiento:

- `train_transaction.csv`
- `train_identity.csv`

> Referencia (a confirmar al incorporar los datos): `train_transaction.csv` contiene aproximadamente 590.000 transacciones. Los datos de test (`test_transaction.csv`, `test_identity.csv`) podrán incorporarse posteriormente.

### 2.2 Variables y entidades relevantes

- **Variable objetivo:** `isFraud` (binaria; 0 = no fraude, 1 = fraude). Su definición y distribución deben confirmarse al cargar los datos.
- **Clave de unión:** `TransactionID` une ambos datasets. No todas las transacciones tienen registros de identity.
- **Clave temporal:** `TransactionDT`, un timedelta en segundos desde una fecha de referencia (no un timestamp real). Al cargar los datos debe confirmarse su rango y comportamiento.
- **Entidades relevantes** a identificar para análisis y feature engineering:
  - transacción;
  - tarjeta (columnas `card*`);
  - dirección / geografía (`addr*`, `dist*`);
  - dispositivo / cliente (columnas de identity: `DeviceType`, `DeviceInfo`, `id_*`);
  - dominio de email (`P_emaildomain`, `R_emaildomain`).
- Columnas de referencia confirmadas en el dataset (a validar al cargar): `TransactionAmt`, `ProductCD`, `card1–card6`, `addr1/addr2`, `M1–M9`, `C1–C14`, `D1–D15`, `V1–V339`, `id_01–id_38`.

### 2.3 Tratamiento de datos raw

- Los archivos raw deben preservarse sin modificaciones.
- Todo análisis y transformación debe realizarse sobre copias derivadas.
- La estructura real de columnas, tipos y misses debe confirmarse mediante inspección del archivo antes de fijar cualquier supuesto.

### 2.4 Requisitos de validación y calidad

Antes de usar los datos debe verificarse:

- lectura correcta de ambos archivos;
- consistencia del tipo de `TransactionID` en ambos datasets;
- cobertura del join entre transaction e identity;
- tipos de datos por columna;
- proporción y patrón de missing values;
- rango y unicidad esperados de `TransactionID`;
- distribución de la variable objetivo.

### 2.5 Separación desarrollo / evaluación

- Los datos de entrenamiento deben dividirse en conjuntos de desarrollo y evaluación (ver §6).
- El test debe mantenerse exclusivo de la evaluación final.

## 3. Data understanding y EDA

Requisito obligatorio: realizar un análisis exploratorio que documente:

- tipos de variables (numéricas, categóricas, binarias);
- missing values por columna y su patrón;
- cardinalidad de variables categóricas;
- distribuciones (univariadas y bivariadas relevantes);
- distribución de fraude y desbalance de clases;
- comportamiento temporal de `TransactionDT` y de la fraude a lo largo del tiempo;
- relaciones relevantes entre variables y con la variable objetivo;
- posibles anomalías o inconsistencias de datos.

> Referencia (a confirmar): la tasa de fraude en entrenamiento es aproximadamente del 3,5 %, lo que implica un fuerte desbalance de clases.

El EDA debe producir conclusiones escritas que sirvan de fundamento para las decisiones de preprocessing y feature engineering.

## 4. Preprocessing

- Construir un pipeline de preprocessing reproducible, ejecutable como unidad.
- Las transformaciones que determine el análisis deben contemplar, sin fijarse de antemano:
  - tratamiento de missing values;
  - variables categóricas y su encoding;
  - variables numéricas y scaling cuando sea necesario;
  - eliminación o conservación justificada de variables;
  - manejo de valores raros o atípicos solo si el EDA lo justifica.
- No fijar transformaciones específicas antes de que el EDA las justifique.
- El pipeline debe serializarse o versionarse para reutilizarse de forma consistente durante inferencia.

## 5. Feature engineering

- Implementar un componente específico y separado de feature engineering.
- El componente debe permitir crear features relacionadas con:
  - tiempo (ciclos, horas, días relativos, etc.);
  - frecuencia de transacciones;
  - comportamiento histórico del cliente o tarjeta;
  - importes y sus transformaciones;
  - entidades y combinaciones de entidades;
  - relaciones entre entidades;
  - comportamiento relativo de una transacción respecto al historial.
- Cada feature relevante debe documentarse indicando:
  - qué representa;
  - cómo se calcula;
  - qué información utiliza;
  - si está disponible en el momento de inferencia.
- Debe prestarse atención especial a las features temporales y a las agregaciones históricas, verificando que no introduzcan información futura.

## 6. División de datos

- Definir una estrategia de train/validation/test adecuada para datos transaccionales y temporales.
- La división debe impedir que información futura influya en el entrenamiento.
- Debe quedar especificado:
  - cómo se generan los tres conjuntos;
  - en qué momento se usa cada uno;
  - cómo se mantiene aislado el test;
  - cómo se garantiza que la división sea reproducible (semilla y/o criterio temporal documentado).

## 7. Modelado

- Estrategia progresiva, en etapas:
  1. modelo baseline simple;
  2. modelos clásicos de referencia;
  3. modelos de boosting;
  4. tuning de hiperparámetros cuando las etapas anteriores lo justifiquen.
- El sistema debe permitir comparar experimentos bajo condiciones equivalentes (misma división, mismas features, misma evaluación).
- No asumir que un modelo específico será el definitivo.

## 8. Evaluación

### 8.1 Conjunto mínimo obligatorio de métricas

- **ROC-AUC** (discriminación general).
- **PR-AUC** (adecuada para clases desbalanceadas).
- **Recall en un threshold operativo** fijado según requerimiento de negocio.
- **Matriz de confusión** en el threshold seleccionado.

### 8.2 Requisitos de evaluación

- Evaluar también o reportar precision, F1 y ROC-AUC como métricas secundarias de contexto.
- Analizar diferentes thresholds de clasificación y elegir el operativo de forma justificada.
- El test set debe utilizarse únicamente para la evaluación final.
- Pueden utilizarse como complemento: ROC/PR curves, y sensibilidad de las métricas al threshold.

## 9. Explainability

- Permitir analizar por qué el modelo realiza sus predicciones:
  - feature importance global;
  - SHAP u otra técnica equivalente.
- La solución debe permitir estudiar el comportamiento global del modelo y, cuando sea posible, predicciones individuales.
- Los artefactos de explainability deben quedar asociados al experimento correspondiente.

## 10. Experiment tracking

- Integrar **MLflow** para registrar:
  - experimentos;
  - parámetros;
  - métricas;
  - modelos;
  - artefactos relevantes (curvas, matrices de confusión, reportes SHAP, features usadas).
- Los experimentos registrados deben poder reproducirse y compararse entre sí.

## 11. Inferencia

- Implementar una etapa de inferencia que:
  - reciba una nueva transacción;
  - aplique exactamente las transformaciones del pipeline de entrenamiento;
  - genere una probabilidad de fraude;
  - aplique un threshold configurable;
  - devuelva una predicción interpretable (clase y/o riesgo) junto con la probabilidad.
- La lógica de preprocessing y feature engineering debe ser exactamente la misma en entrenamiento e inferencia.

## 12. API y empaquetado

- Evolución posterior: exponer la inferencia mediante una API con una tecnología apropiada (por ejemplo **FastAPI**).
- La API debe permitir:
  - recibir los datos de una transacción;
  - ejecutar preprocessing y feature engineering;
  - ejecutar el modelo;
  - devolver resultado y probabilidad.
- La implementación concreta puede definirse durante las tareas correspondientes.

## 13. Docker y MLOps

- Evolución posterior del proyecto:
  - Docker;
  - tests automatizados;
  - CI/CD;
  - MLflow (servicio/registry);
  - monitoring.
- No es necesario implementar estos componentes inicialmente.
- Cada componente debe incorporarse cuando las etapas anteriores estén suficientemente estables.

## 14. Estructura conceptual

- El sistema completo debe evolucionar aproximadamente según:

`Raw Data → Validation → EDA → Preprocessing → Feature Engineering → Split → Training → Evaluation → Explainability → MLflow → Inference → API → Docker → Monitoring`

- La estructura final de carpetas y archivos no debe fijarse arbitrariamente en este documento mientras no se haya tomado esa decisión.

## 15. Criterios de aceptación

Criterios verificables por etapa:

- **Datos validables:** ambos archivos se leen correctamente, el join por `TransactionID` es consistente y la calidad está documentada.
- **Preprocessing reproducible:** el pipeline se ejecuta de forma idéntica y reproducible al menos dos veces sobre los mismos datos.
- **Features consistentes:** las features pueden generarse de forma consistente en entrenamiento e inferencia.
- **Sin leakage conocido:** se comprueba que ninguna feature utiliza información futura o indisponible en inferencia.
- **Entrenamiento reproducible:** el modelo puede entrenarse y reproducirse con resultados comparables.
- **Métricas calculables:** el conjunto mínimo de métricas (§8.1) se calcula sin errores sobre los conjuntos correspondientes.
- **Experimentos registrados:** cada experimento relevante queda registrado en MLflow con su código, datos, configuración y resultados asociados.
- **Pipeline completo:** una nueva transacción atraviesa el pipeline de principio a fin sin errores.
- **Inferencia consistente:** la inferencia utiliza el mismo procesamiento que el entrenamiento, verificable mediante el pipeline compartido.