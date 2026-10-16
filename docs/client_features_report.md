# Reporte de identidad de cliente (T-E02)

## Por qué esta tarea

El objetivo de negocio de **recall ≥ 0,80 no se cumplió** en `test` (recall 0,7707, IC 95 % `[0,7551, 0,7864]`, que excluye 0,80). La ROC-AUC quedó en **0,9025**.

El ganador de la competencia IEEE-CIS Fraud Detection obtuvo **0,9459** en el leaderboard privado. La diferencia no es de modelo: el ganador usó ensemble de XGBoost + CatBoost + LightGBM sobre ~262 features, y nosotros LightGBM sobre 764 columnas, de las cuales solo 20 son features propias. Ese desequilibrio es la hipótesis principal, y este reporte la acota con datos de **nuestro** split, no con la receta ajena.

La feature que la referencia describe como decisiva fue un **UID de cliente**: `card1 + addr1 + D1n`, con `D1n = TransactionDay - D1` (el día en que arrancó la tarjeta), usado como clave de agregación sobre las columnas C y M.

Antes de copiar esa receta, se midió si funciona con nuestro split. **No directamente**, y el plan se ajusta a lo que los datos muestran.

## Hallazgo 1: la etiqueta es de cliente, no de transacción

La referencia original definió la etiqueta a nivel de cuenta: cuando una tarjeta registra fraude, la organización marca como fraudulentas las transacciones posteriores vinculadas a esa tarjeta. Se verificó en nuestros datos agrupando `train` por `card1 + addr1 + D1n`:

| Métrica | Valor | Referencia |
|---|---|---|
| Grupos (UID) en `train` | 170.598 | 73.838 clientes |
| UID de una sola transacción | 99.399 (58,3 %) | — |
| UID con etiqueta 100 % fraude | 3.757 (2,2 %) | 2,9 % |
| UID con **etiqueta mixta** | **2.346 (1,4 %)** | **0,2 %** |

El patrón cualitativo se confirma: la etiqueta es esencialmente de cliente, y muy pocos grupos mezclan clases. Tratar cada transacción como independiente ignora esa estructura.

## Hallazgo 2: la actividad del cliente es una señal fuerte y monótona

Tasa de fraude en `train` según cuántas transacciones tiene el cliente:

| Transacciones del cliente | Grupos | Tasa de fraude |
|---|---|---|
| 1 | 99.399 | 2,29 % |
| 2 | 28.462 | 2,97 % |
| 3–5 | 26.711 | 3,17 % |
| 6–10 | 10.716 | 3,62 % |
| 11–50 | 5.175 | 4,60 % |
| 50+ | 135 | **10,04 %** |

Tasa global de `train`: 3,51 %. La relación es monótona y va de 2,29 % a 10,04 %, un factor 4,4. Es señal explotable, y `cnt_card1_addr1` ya la captura parcialmente.

## Hallazgo 3: el UID de la referencia no sirve tal cual en nuestro split

Esto es lo que cambia el plan. Se midió cuántos valores de cada clave aparecen alguna vez en `train` (es decir, una fila de test tiene historial de esa entidad):

| Clave | Valores distintos en `train` | Valores nuevos en `test` | **Filas de test con historial** |
|---|---|---|---|
| `card1` | 12.421 | 674 | **98,7 %** |
| `addr1` | 322 | 2 | 97,6 % |
| `card1+addr1` | 35.384 | 2.585 | **95,1 %** |
| `card1+addr1+D1n` | 170.598 | 26.981 | **34,4 %** |

Por uniques, el UID de la referencia es 9 veces más fino que `card1+addr1`; por cobertura de test, **65,6 % de las filas se quedarían sin valor agregado alguno**.

La razón es visible en la naturaleza del UID: `D1n` es el día de alta de la tarjeta, así que distingue un stint de uso concreto del mismo plástico. En la competencia el 68,2 % de los clientes del test privado eranenteramente nuevos, y la referencia lo dice explícitamente: no se podía usar el UID como feature por eso mismo, solo como clave de agregación. En nuestro split temporal la proporción de es mucho mayor porque el plástico se reutiliza entre periodos.

**Conclusión:** la clave primaria es `card1+addr1` (95,1 %), y el UID completo se conserva como **segundo nivel** para el 34,4 % de filas que sí tienen historial de cliente exacto. Se aplica degradación natural: si `D1` es nulo, la fila solo recibe el nivel primario.

## Hallazgo 4: las columnas C, M y D ya están disponibles

El parquet actual ya contiene **339 V, 14 C, 9 M y 14 D**, más 366 indicadores de missingness. Lo que faltaba no eran columnas sino **agregaciones por entidad**. `D1` está presente en crudo, pero ninguna feature derivada de las columnas D existe en el catálogo actual.

## Hallazgo 5 (fuera de alcance): identificadores tratados como float

`card1`, `addr1`, `card2`, `card3`, `addr2`, `D1`, `P_emaildomain` y `ProductCD` están persistidas como `float`. LightGBM las trata como continuas y corta por umbrales numéricos sobre identificadores que no admiten orden: un umbral de `card1` no significa nada. Nunca se declararon como categóricas.

Es un cambio de una línea por columna y probablemente de impacto comparable a T-E02, pero es una decisión de modelado distinta y queda registrada como pendiente, no mezclada en esta tarea.

## Diseño acordado

| Aspecto | Decisión | Motivo |
|---|---|---|
| Clave primaria | `card1 + addr1` | 95,1 % de cobertura en test |
| Clave refinada | `card1 + addr1 + D1n` | Solo aporta en el 34,4 % con historial |
| Columnas | C1–C14 | M1–M9 son categóricas en crudo, no admiten "media previa" |
| Agregación | Media previa, acumulativa | Fiel a la referencia y simple (`constitution.md` §7) |
| `D1n` | Clave interna, no feature | Su información ya está en `D1`; emitirlo reintroduce el identificador que la referencia rechaza |
| Sin historial | Centinela `0.0` | Nunca `NaN`: rompe `test_inferencia_fila_unica_consistente` y el check `.notna()` del pipeline |

### Fórmula vectorizada

Con el frame ya ordenado por `TransactionDT`, para cada columna `col` de C y cada clave `keys`:

```
x        = col.fillna(0.0)
nonnull  = col.notna()
prior_sum  = groupby(keys)[x].cumsum() - x
prior_n    = groupby(keys)[nonnull].cumsum() - nonnull
prior_mean = prior_sum / prior_n   si prior_n > 0, si no 0.0
```

`cumsum() - x` da la suma de filas **previas** del grupo sin necesidad de `shift`, y es correcto porque la acumulación por grupo respeta el orden temporal del frame. Los `NaN` de C se tratan como 0 en el numerador y no cuentan en el denominador, de modo que una columna ausente en una fila no destruye el agregado del grupo.

El atajo importa: la definición ingenua (`groupby.transform` con `expanding().mean().shift(1)`) haría ~35.384 × 14 = 495.000 llamadas de Python por clave. Por eso `test_media_vectorizada_coincide_con_expanding` es el test crítico de esta tarea: ata la optimización a la definición para que no puedan divergir.

### Features resultantes

| Familia | Cantidad | Clave |
|---|---|---|
| `cnt_client_uid` | 1 | refinada |
| `mean_C{i}_prev_client` | 14 | primaria |
| `mean_C{i}_prev_uid` | 14 | refinada |
| **Total** | **29** | |

De 20 features propias a 49. Parquet de 764 a 793 columnas.

**Desviación respecto al diseño inicial:** el diseño proyectaba 47 features (C+M en las dos claves). M1–M9 quedaron **excluidas** al implementar, por tres razones medidas: en el dataset crudo son categóricas (`T`/`F`, y M4 con tres niveles `M0`/`M1`/`M2`), de modo que su media previa no está definida sobre el frame que recibe `compute_features` —el preprocessor ya las declara categóricas y las codifica con `OrdinalEncoder`—; M1 es `T` en el 99,99 % de sus no nulos; y entre el 46 % y el 59 % de las M son nulas, luego su tasa previa sería casi constante y calculada sobre historiales muy cortos. Las C sí son recuentos numéricos de coincidencias, que es justo lo que promedia la referencia. El test `test_columnas_m_no_se_agregan` fija esta decisión.

## Regla de certificación

Añadir features es selección de features. `constitution.md` §5 exige que `test` se mantenga independiente de ella, y `test` **ya fue consumido** en la Fase I.

- `run_tuning.py` y `run_threshold.py` se re-ejecutan.
- `run_evaluation.py` **no** se vuelve a ejecutar.

La cifra de `test` — ROC-AUC 0,9025, PR-AUC 0,5282, recall 0,7707 — **corresponde al modelo baseline**. El modelo con T-E02 se certifica con train out-of-fold y `validation`, y debe reportarse así.

La comparación se hace por fold, no solo por media: si las features no mejoran de forma consistente en los 3 pliegues walk-forward, es ruido y T-E02 no entra.

## Resultado medido (2026-09-28)

Ejecutado: `run_feature_engineering.py` (parquet reconstruido, 764 → 793 columnas), `run_tuning.py` (segunda búsqueda, mismas 20 configs y seed 42) y `run_threshold.py`. **`run_evaluation.py` no se ejecutó: `test` no se volvió a leer.** El modelo de Fase H quedó archivado en `artifacts/model_final_fase_h.joblib` y sus métricas de `test` en `reports/evaluation/fase_i_baseline/`.

### Por fold, la regla que el propio plan se impone

| Comparación | PR-AUC OOF | Pliegues | Δ media |
|---|---|---|---|
| Ganadora con T-E02 (config 11, 439 rondas) | 0,640227 | [0,6830, 0,5667, 0,6709] | +0,006654 |
| Ganadora de Fase H (config 18, 219 rondas) | 0,633573 | [0,6727, 0,5557, 0,6723] | — |
| **Δ por pliegue** | | **[+0,0104, +0,0110, −0,0014]** | |
| Ancla con T-E02 (config 0) | 0,630059 | [0,6692, 0,5568, 0,6641] | −0,001596 |
| Ancla de Fase H (config 0) | 0,631655 | [0,6782, 0,5482, 0,6686] | — |
| **Δ por pliegue** | | **[−0,0089, +0,0086, −0,0044]** | |

La meta del plan (OOF > 0,633573) **se cumple**, con +0,0067 de media. Pero la regla escrita en este mismo reporte dice que el cambio entra solo si mejora de forma consistente en los 3 pliegues, y **no lo hace**: el tercer pliegue se va −0,0014. El ruido entre pliegues es de ±0,05, así que la señal está dentro del ruido.

La comparación que aísla las features es la del ancla, porque mantiene los hiperparámetros de la Fase G y solo cambia el set de columnas: **−0,0016 de media, con los tres signos distintos.** Atribuir el +0,0067 a las features de cliente no se sostiene; la diferencia real entre ambas búsquedas es que la seleccionó otra configuración (255 hojas y 439 rondas, frente a 219).

### Confirmación en `validation`

| Métrica | T-E02 | Fase H | Δ |
|---|---|---|---|
| ROC-AUC | 0,927395 | 0,926248 | +0,001147 |
| PR-AUC | 0,607936 | 0,581350 | +0,026586 |
| recall @0,5 | 0,342055 | 0,340538 | +0,001517 |
| precision @0,5 | 0,866475 | 0,843192 | +0,023282 |

Mejora visible, pero **mezcla dos cambios**: las features y la configuración ganadora. No es atribuible a T-E02 aislada.

### Umbral operativo

`run_threshold.py` fijó el umbral en `validation` con el objetivo de recall ≥ 0,80:

| Bloque | Umbral | recall | precision | alertas |
|---|---|---|---|---|
| validation (calibración) | 0,008480 | 0,8002 [0,7849, 0,8154] | 0,2399 | 11,30 % |
| train out-of-fold (corroboración) | 0,008480 | 0,7815 [0,7723, 0,7906] | 0,2544 | 12,01 % |
| _Fase H, validation_ | _0,029329_ | _0,8002_ | _0,2107_ | _12,87 %_ |
| _Fase H, train OOF_ | _0,029329_ | _0,8161_ | _0,2112_ | _15,12 %_ |

El modelo con T-E02 alcanza el mismo recall de validación con **menos alertas y más precisión** que el de Fase H (11,30 % frente a 12,87 %; 0,2399 frente a 0,2107), y en `test` el margen fue precisamente el de las alertas. Es el indicio más favorable a T-E02, y es un indicio de **punto operativo**, no de ranking: no separa mejor los ejemplos.

Cambia el signo de la deriva: con Fase H el recall caía al avanzar en el tiempo (OOF 0,8161 → validation 0,8002 → test 0,7707, deriva −0,0159), y con T-E02 la corroboración out-of-fold queda **por debajo** de la calibración (0,7815 frente a 0,8002, deriva +0,0187). No se puede comprobar si eso revierte la caída en `test` sin volver a consumir `test`.

### Las features sí se usan

Importancia por ganancia del modelo final (791 features, `artifacts/model_final.joblib`):

| Familia | n | Ganancia total | Mejor puesto |
|---|---|---|---|
| Todas las de cliente | 29 | 13,28 % | #10 |
| `mean_C{i}_prev_client` (primaria) | 14 | 7,90 % | #27 |
| `cnt_client_uid` + `mean_C{i}_prev_uid` (refinada) | 15 | 5,38 % | #10 (`cnt_client_uid`, 2,01 %) |

`cnt_client_uid` es la décima feature más usada del modelo, por encima de `card1` o `TransactionAmt`. El modelo las explotó; el problema es que no produce ranking suficiente para superar el ruido entre pliegues. La cola de la familia refinada es floja: `mean_C3_prev_uid` queda en el puesto 399 de 791 con 0,002 % de ganancia.

### La cobertura de las claves no es el problema

Descartada la hipótesis de "la clave refinada casi nunca aplica". Cobertura de la clave en la ventana de puntuación de cada pliegue de `train` (la fila ya tiene historial en la ventana de entrenamiento):

| Pliegue | n puntuadas | `card1+addr1` | `+D1n` |
|---|---|---|---|
| 0 | 63.662 | 94,0 % | 41,5 % |
| 1 | 71.089 | 94,4 % | 40,6 % |
| 2 | 65.544 | 95,7 % | 44,8 % |

La cobertura es estable y alta en los tres pliegues, incluido el tercero, que es el que no mejora. `D1` no es el cuello de botella.

## Veredicto

**T-E02 no se certifica como mejora**, y por la regla escrita arriba (consistencia en los 3 pliegues) no debería entrar como cambio aceptado de feature engineering. Queda implementado, testeado y medido, con un resultado negativo que es información valiosa por sí mismo: la identidad de cliente es la vía que la referencia de la competencia siguió para llegar a 0,946 y aquí no mueve el ranking.

Lo que sí es defendible del cambio, con las cifras delante:

- El punto operativo mejora: mismo recall de validación con 1,57 puntos menos de alertas (11,30 % frente a 12,87 %) y precisión 0,2399 frente a 0,2107. Es la única comparación que favorece a T-E02 sin estar contaminada por la configuración elegida.
- El drift entre bloques cambia de signo, lo que sugiere —sin poder comprobarlo— un comportamiento temporal menos degradado.
- La comparación aislada (ancla, mismos hiperparámetros) es **negativa**: −0,0016 de media y tres signos distintos. La explicación más honesta de la diferencia entre las dos búsquedas es la configuración, no las features.

Pendiente de decisión: si se conserva el modelo de Fase H como desplegable o se adopta el de T-E02, y si `test` —ya consumido— llega a hacer falta para dirimirlo, porque sin un holdout limpio no hay forma de resolverlo con rigor.

## Riesgos planteados y cómo quedaron

- **Sobreajuste.** 29 features sobre 434k filas con 439 rondas. No se materializó: el desglose por pliegue no muestra caída en los pliegues recientes respecto a la línea base, y las features reciben 13,3 % de la ganancia.
- **Cobertura de `D1`.** Descartada por medición: 40,6–44,8 % de las filas puntuadas de cada pliegue tienen historial de cliente exacto, estable entre pliegues. Ver la tabla de arriba.
- **Objetivo equivocado.** Planteado como el riesgo principal y confirmado: el ranking no mejora, pero el punto operativo sí (menos alertas al mismo recall). Es exactamente el caso en que una mejora de AUC no se traduce en el objetivo de negocio, y aquí ni siquiera hay mejora de AUC.
- **Deriva temporal.** El recall bajó 0,8161 → 0,8002 → 0,7707 entre train OOF, validation y test en Fase H. Con T-E02 la deriva out-of-fold cambia de signo, pero eso no se puede verificar sin `test`.

## Reproducción del análisis

Las cifras de cobertura salen de `data/processed/split.parquet` con el digest `6d612a6ad9c4328088a91fc4278cd4b3d890e73fa44769df27746f0282c97e6a` de la Fase H. Las de modelado salen de la segunda ejecución de `run_tuning.py` y `run_threshold.py`, con el `split.parquet` reconstruido por T-E02 (digest `162c566a169f740f...`); los resultados anteriores están archivados en `reports/tuning/baseline_fase_h/` y `reports/evaluation/fase_i_baseline/`.
