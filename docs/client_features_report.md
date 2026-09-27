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
| Columnas | C1–C14 + M1–M9 | Baja cardinalidad, medias interpretables |
| Agregación | Media previa, acumulativa | Fiel a la referencia y simple (`constitution.md` §7) |
| `D1n` | Clave interna, no feature | Su información ya está en `D1`; emitirlo reintroduce el identificador que la referencia rechaza |
| Sin historial | Centinela `0.0` | Nunca `NaN`: rompe `test_inferencia_fila_unica_consistente` y el check `.notna()` del pipeline |

### Fórmula vectorizada

Con el frame ya ordenado por `TransactionDT`, para cada columna `col` de C/M y cada clave `keys`:

```
x        = col.fillna(0.0)
nonnull  = col.notna()
prior_sum  = groupby(keys)[x].cumsum() - x
prior_n    = groupby(keys)[nonnull].cumsum() - nonnull
prior_mean = prior_sum / prior_n   si prior_n > 0, si no 0.0
```

`cumsum() - x` da la suma de filas **previas** del grupo sin necesidad de `shift`, y es correcto porque la acumulación por grupo respeta el orden temporal del frame. Los `NaN` de C/M se tratan como 0 en el numerador y no cuentan en el denominador, de modo que una columna ausente en una fila no destruye el agregado del grupo.

El atajo importa: la definición ingenua (`groupby.transform` con `expanding().mean().shift(1)`) haría ~35.384 × 23 = 813.000 llamadas de Python por clave. Por eso `test_media_vectorizada_coincide_con_expanding` es el test crítico de esta tarea: ata la optimización a la definición para que no puedan divergir.

### Features resultantes

| Familia | Cantidad | Clave |
|---|---|---|
| `cnt_client_uid` | 1 | refinada |
| `mean_C{i}_prev_client` | 14 | primaria |
| `mean_M{i}_prev_client` | 9 | primaria |
| `mean_C{i}_prev_uid` | 14 | refinada |
| `mean_M{i}_prev_uid` | 9 | refinada |
| **Total** | **47** | |

De 20 features propias a 67. Parquet de 764 a 811 columnas.

## Regla de certificación

Añadir features es selección de features. `constitution.md` §5 exige que `test` se mantenga independiente de ella, y `test` **ya fue consumido** en la Fase I.

- `run_tuning.py` y `run_threshold.py` se re-ejecutan.
- `run_evaluation.py` **no** se vuelve a ejecutar.

La cifra de `test` — ROC-AUC 0,9025, PR-AUC 0,5282, recall 0,7707 — **corresponde al modelo baseline**. El modelo con T-E02 se certifica con train out-of-fold y `validation`, y debe reportarse así.

La comparación se hace por fold, no solo por media: si las 47 features no mejoran de forma consistente en los 3 pliegues walk-forward, es ruido y T-E02 no entra.

## Qué haría que esto no funcione

- **Sobreajuste.** 47 features sobre 434k filas con 219 rondas. Mitigado por la validación por fold, no por intuición.
- **Cobertura de `D1`.** Las filas con `D1` nulo degradan al nivel primario. Hay que medir cuántas son y reportarlo; si son muchas, la clave refinada aporta poco.
- **Objetivo equivocado.** Si el objetivo de negocio sigue siendo recall a umbral fijo y no AUC, mejorar el ranking puede no mover el recall operativo. Es un riesgo real y no se puede resolver sin volver a medir, lo cual exige un holdout que ya no existe.
- **Deriva temporal.** El recall bajó 0,8161 → 0,8002 → 0,7707 entre train OOF, validation y test. Un modelo mejor puede seguir sin alcanzar 0,80 en producción.

## Reproducción del análisis

Las cifras de este reporte salen de `data/processed/split.parquet` y son reproducibles con el mismo digest `6d612a6ad9c4328088a91fc4278cd4b3d890e73fa44769df27746f0282c97e6a` de la Fase H.
