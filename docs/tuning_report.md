# Tuning y selección de modelo — Fase H

Fecha: 2026-09-26. Módulos: `src/modeling/space.py`, `src/modeling/tuning.py`; ejecución: `run_tuning.py`. Salidas (gitignored): `reports/tuning/search.csv`, `reports/tuning/search.json`, `reports/tuning/selected.json`, `artifacts/model_final.joblib`.

## 1. Alcance

Ajustar hiperparámetros de LightGBM y **seleccionar el modelo definitivo** bajo un criterio declarado, sin que el `test` participe de ninguna decisión (`constitution.md` §5, `specs.md` §7, `tasks.md` T-H01). Esta fase **no** evalúa el `test` ni elige el umbral operativo: eso es Fase I.

## 2. Contrato de uso de datos (specs §6, constitution §5)

| conjunto | filas | uso en Fase H |
|---|---:|---|
| `train` | 434.176 | Ajuste y **selección** (3 pliegues walk-forward internos) |
| `validation` | 77.822 | **Una única confirmación** del modelo ya elegido |
| `test` | 78.542 | **No se carga.** `load_splits` recibe `EVAL_SETS=(train, validation)` y `run_tuning.py` verifica con `assert` que `test` no está presente |

`tune_one` recibe **solo** `train`: la función no tiene forma de alcanzar `validation` o `test`. `selected.json` registra `test_usado: false` y el SHA-256 de `split.parquet` (`6d612a6a…`).

**Desviación respecto de `tasks.md`:** la descripción de T-H01 dice "tuning sobre validation". Se siguió el criterio más estricto de `constitution.md` §5 y el verificador de `load_splits`, que ya reservaban `validation` como conjunto de confirmación: la búsqueda corre sobre pliegues internos de `train` y `validation` se toca una vez, al final. `validation` no interviene en ninguna elección entre configuraciones.

## 3. Protocolo de búsqueda

| elemento | valor | motivo |
|---|---|---|
| modelo | LightGBM, `objective=binary` | elegido en Fase G sobre evidencia |
| muestreo | 20 configuraciones aleatorias, `seed=42` | `constitution.md` §7: no introducir `optuna` sin razón |
| configuración 0 | ancla de Fase G (`num_leaves=63`, `lr=0.05`, `min_child=100`, `colsample=0.6`, `subsample=0.8`, `reg_lambda=1.0`, `spw=1.0`) | punto de comparación; permite afirmar "el tuning mejoró/no mejoró" |
| evaluación | 3 pliegues walk-forward de `train` (§5 de `docs/modeling_report.md`) | nunca se evalúa sobre el mismo periodo en que se ajusta |
| criterio de selección | **PR-AUC medio de los 3 pliegues** | clase positiva al 3,5 %; ROC-AUC no discrimina bien el desbalance |
| desempate | ROC-AUC medio, luego `min_pr_auc`, luego `config_id` | criterio total y determinista |
| rondas | techo 600, early stopping 30 rondas sobre `average_precision` | el corte lo fija la métrica de selección, no un número arbitrario |
| paridad | `deterministic=True`, `force_row_wise=True`, `num_threads=-1` | replicabilidad entre hardware y número de hilos |

Espacio muestreado (`src/modeling/space.py`), con rangos derivados del EDA y de la Fase G:

| hiperparámetro | valores |
|---|---|
| `num_leaves` | 15, 31, 63, 127, 255 |
| `learning_rate` | 0,05 / 0,10 / 0,20 |
| `min_child_samples` | 20 / 100 / 500 / 2000 |
| `colsample_bytree` | 0,4 / 0,6 / 0,8 / 1,0 |
| `subsample` | 0,7 / 0,8 / 1,0 |
| `reg_lambda` | 0 / 1 / 10 / 100 |
| `reg_alpha` | 0 / 0,1 / 1,0 |
| `scale_pos_weight` | 1,0 / 27,5 |

El muestreo sesgó deliberadamente la dimensión de compensación de clases: **11 configuraciones con `scale_pos_weight=1.0` y 9 con `27.5`**, para que esa decisión se tomara con evidencia y no por el azar del muestreo. El reparto exacto de las 20 configuraciones está en `search.csv`; el desbalance 11/9 es consecuencia del muestreo aleatorio, no de un diseño balanceado.

## 4. Corrección de un defecto metodológico

La primera ejecución de la búsqueda **produjo resultados inválidos y se descartó**. Queda registrado porque el fallo era invisible en los outputs y fácil de repetir.

**Síntoma.** 9 de las 10 configuraciones con `scale_pos_weight=27.5` terminaban en `best_iteration=1` con PR-AUC de 0,18–0,33, y quedaban todas por debajo de las de `spw=1.0`. La conclusión evidente —"compensar el desbalance arruina el ranking"— era un artefacto del protocolo.

**Causa.** `tune_one` construía los parámetros con `build_params`, que **no fija la métrica**. Sin `metric` explícita, LightGBM evalúa con su métrica por defecto, `binary_logloss`, y el early stopping la minimizaba. Con `scale_pos_weight` alto, el logloss se minimiza casi en la primera ronda (el modelo queda inmediatamente sobrediñado en la clase positiva) y el corte ocurre antes de que el modelo haya aprendido a ordenar. La constante `EARLY_STOPPING_METRIC` existía y un test comprobaba su valor, pero **nada la conectaba con LightGBM**: el test pasaba de forma vacía.

**Diagnóstico.** Con la métrica fijada explícitamente, la curva de `average_precision` reportada por LightGBM **coincide con `average_precision_score` de scikit-learn hasta la última cifra decimal en 1, 20 y 80 rondas**, y crece de forma monótona. Para `spw=27.5` sobre el pliegue 0: PR-AUC 0,2826 → 0,6402 entre la ronda 1 y la 400, sin meseta. El modelo nunca se había estabilizado; se lo cortaba.

**Corrección.** Se añadió `build_search_params()`, que inyecta `metric=EARLY_STOPPING_METRIC` en los parámetros de búsqueda, y se usa en `tune_one`. `fit_final` sigue usando `build_params` sin métrica (no hay conjunto de evaluación al ajustar el modelo definitivo). Se añadieron dos tests no vacíos: uno que verifica la coincidencia LightGBM↔scikit-learn de PR-AUC en distintas rondas, y otro que comprueba que la métrica registrada es `average_precision` y no la por defecto. Aprovechando la corrección se redujo además el techo de rondas de 1000 a **600**, para acotar coste y mantener el rango comparable con la referencia de Fase G (600 rondas sin early stopping).

**Efecto.** En datos reales, la misma configuración con `spw=27.5` pasó de `best_iteration=1` / PR-AUC ≈0,18 a `best_iteration=353` / PR-AUC 0,6061. La búsqueda se re-ejecutó completa desde cero (mismas 20 configuraciones, mismo `seed=42`: la única variable distinta fue la métrica). Todos los números de este documento son de la ejecución corregida.

## 5. Resultados de la búsqueda (60 ajustes, 87,0 min)

Ordenado por PR-AUC medio. `iters` es la mediana de la ronda de early stopping por pliegue.

| # | num_leaves | lr | min_child | colsample | subsample | reg_λ | reg_α | spw | **PR-AUC** | std | min | ROC-AUC | recall@0,5 | iters | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 127 | 0,05 | 100 | 0,6 | 1,0 | 0,0 | 0,0 | 1,0 | **0,6336** | 0,055 | 0,5557 | 0,9243 | 0,419 | 219 | 417,8 |
| 0 | 63 | 0,05 | 100 | 0,6 | 0,8 | 1,0 | 0,0 | 1,0 | 0,6317 | 0,059 | 0,5482 | 0,9226 | 0,422 | 470 | 153,8 |
| 11 | 255 | 0,10 | 20 | 1,0 | 0,8 | 10,0 | 0,0 | 1,0 | 0,6297 | 0,057 | 0,5502 | 0,9208 | 0,402 | 247 | 300,3 |
| 4 | 127 | 0,10 | 100 | 1,0 | 0,8 | 1,0 | 0,1 | 1,0 | 0,6268 | 0,064 | 0,5368 | 0,9195 | 0,414 | 272 | 174,3 |
| 12 | 127 | 0,20 | 100 | 1,0 | 1,0 | 1,0 | 1,0 | 1,0 | 0,6213 | 0,056 | 0,5417 | 0,9154 | 0,431 | 441 | 317,7 |
| 15 | 63 | 0,10 | 500 | 0,8 | 0,7 | 0,0 | 0,0 | 1,0 | 0,6211 | 0,062 | 0,5330 | 0,9163 | 0,437 | 270 | 289,9 |
| 9 | 63 | 0,05 | 2000 | 0,6 | 0,7 | 1,0 | 1,0 | 1,0 | 0,6178 | 0,056 | 0,5390 | 0,9164 | 0,417 | 502 | 193,4 |
| 17 | 63 | 0,10 | 500 | 0,8 | 0,7 | 10,0 | 1,0 | 1,0 | 0,6167 | 0,056 | 0,5373 | 0,9163 | 0,416 | 280 | 348,0 |
| 19 | 255 | 0,20 | 20 | 0,4 | 1,0 | 0,0 | 1,0 | 1,0 | 0,6084 | 0,056 | 0,5297 | 0,9130 | 0,427 | 330 | 545,0 |
| 8 | 63 | 0,05 | 500 | 0,8 | 1,0 | 10,0 | 0,1 | 27,5 | 0,6080 | 0,055 | 0,5307 | 0,9161 | **0,685** | 523 | 180,8 |
| 7 | 127 | 0,20 | 2000 | 0,4 | 0,8 | 1,0 | 0,1 | 1,0 | 0,6066 | 0,057 | 0,5256 | 0,9107 | 0,427 | 189 | 115,4 |
| 10 | 63 | 0,05 | 500 | 0,6 | 0,7 | 0,0 | 0,1 | 27,5 | 0,6058 | 0,055 | 0,5286 | 0,9165 | **0,695** | 371 | 146,7 |
| 3 | 63 | 0,05 | 2000 | 0,6 | 0,8 | 1,0 | 0,0 | 27,5 | 0,6033 | 0,056 | 0,5238 | 0,9180 | **0,701** | 504 | 204,2 |
| 13 | 31 | 0,20 | 500 | 0,4 | 1,0 | 0,0 | 1,0 | 1,0 | 0,6027 | 0,059 | 0,5194 | 0,9070 | 0,421 | 223 | 202,2 |
| 14 | 127 | 0,20 | 2000 | 0,8 | 0,8 | 10,0 | 0,0 | 27,5 | 0,6005 | 0,061 | 0,5147 | 0,9061 | 0,550 | 455 | 640,1 |
| 2 | 31 | 0,05 | 500 | 1,0 | 1,0 | 100,0 | 1,0 | 27,5 | 0,5981 | 0,053 | 0,5235 | 0,9131 | **0,728** | 479 | 149,3 |
| 16 | 63 | 0,20 | 500 | 0,6 | 1,0 | 10,0 | 0,0 | 27,5 | 0,5963 | 0,058 | 0,5192 | 0,9023 | 0,575 | 600 | 473,7 |
| 5 | 15 | 0,10 | 2000 | 0,4 | 1,0 | 100,0 | 0,0 | 27,5 | 0,5841 | 0,046 | 0,5193 | 0,9095 | **0,740** | 464 | 141,4 |
| 1 | 15 | 0,20 | 500 | 0,6 | 0,8 | 100,0 | 0,0 | 27,5 | 0,5807 | 0,060 | 0,4958 | 0,9061 | **0,721** | 405 | 100,1 |
| 6 | 15 | 0,20 | 500 | 0,6 | 0,7 | 100,0 | 0,1 | 27,5 | 0,5727 | 0,055 | 0,4944 | 0,9016 | **0,747** | 277 | 78,1 |

### Desglose por pliegue de las dos primeras

| configuración | pliegue 0 (sem 9–11) | pliegue 1 (sem 12–14) | pliegue 2 (sem 15–17) |
|---|---:|---:|---:|
| 18 (ganadora) | 0,6727 (154 rondas) | 0,5557 (219) | 0,6723 (393) |
| 0 (ancla) | 0,6782 (470) | 0,5482 (300) | 0,6686 (598) |

## 6. Lectura

1. **El tuning no produjo una mejora real.** La ganadora supera al ancla en **+0,0019** de PR-AUC medio (0,6336 vs 0,6317). La dispersión entre pliegues de la misma configuración es de **±0,046 a ±0,064** — entre **24 y 34 veces** el margen. Las dos configuraciones son estadísticamente indistinguibles con este diseño. Se mantiene la ganadora porque el criterio de §3 estaba declarado antes de ver los resultados y no se cambia a posteriori: reescoger el ancla por ser "más simple" sería seleccionar con información que el criterio no contemplaba.
2. **La configuración de Fase G ya estaba casi en el óptimo.** El resultado honesto de esta fase es **negativo en cuanto a ganancia**: la referencia de la Fase G se sitúa en el segundo puesto de 20. El aporte de la Fase H es haberlo mostrado con evidencia, y haber descartado explícitamente 19 alternativas.
3. **Confirmado con datos el compromiso de la Fase G sobre el desbalance.** Los **nueve primeros puestos son los nueve mejores `scale_pos_weight=1.0`** (PR-AUC 0,6027–0,6336, recall@0,5 de 0,40–0,44). Las 9 configuraciones con `27.5` arrancan en el puesto 10 y cierran la tabla (0,5727–0,6080), a cambio de alcanzar 0,55–0,75 de recall. **El solapamiento es de 4 puestos** (10–14): la mejor configuración compensada (id 8, 0,6080) supera a la peor no compensada (id 13, 0,6027). La compensación no recupera ranking de forma consistente, y su ventaja en recall es consecuencia directa de predecir positivo más a menudo, no de un modelo mejor. Es el mismo punto de la curva que medía `lgbm_pos_weight` en la Fase G, ahora con 9 configuraciones en lugar de una.

4. **El peso de cada hiperparámetro, por promedios marginales.** Agrupando las 20 configuraciones por valor (con la salvedad de §9: el diseño no es un factorial completo, así que los marginales están confundidos y no aíslan el efecto de un hiperparámetro):

| hiperparámetro | señal observada |
|---|---|
| `scale_pos_weight` | el más fuerte: media 0,6197 (`1.0`) frente a 0,5944 (`27.5`) |
| `reg_lambda` | `100` es el peor grupo (media 0,5839, puestos 16, 18, 19 y 20); `0` y `1` son los mejores (0,614 / 0,618) |
| `learning_rate` | `0.20` es el peor grupo (0,5987); `0.05` y `0.10` empatan (0,614 / 0,616) |
| `num_leaves` | monótono: `15` da 0,5792 (puestos 18–20), `255` da 0,6191 |
| `min_child_samples` | `100` es el mejor grupo (0,6283, puestos 1, 2, 4 y 5); `500` y `2000` caen a 0,600 / 0,603 |
| `colsample_bytree` | señal débil y no monótona: `0.4` peor (0,6005), `1.0` mejor (0,6190) |
| `subsample` | sin señal: 0,6066 / 0,6113 / 0,6068 |
| `reg_alpha` | sin señal: 0,6090 / 0,6040 / 0,6108 |

Tres consecuencias. Primera: **el óptimo está en el centro del espacio**, no en un extremo — la regularización fuerte sobra y `min_child_samples` alto también penaliza. Segunda: **la ganadora no es la mejor en ninguna dimensión salvo `num_leaves`**; sus `reg_alpha=0.0` y `subsample=1.0` no reflejan una ventaja medida, sino ruido de muestreo dentro de un grupo sin señal. Tercera: `colsample_bytree`, `subsample` y `reg_alpha` **podrían fijarse** en una iteración futura: no aportaron señal y multiplican el espacio de 480 a 17.280 combinaciones.

5. **El pliegue 1 es sistemáticamente más difícil** (PR-AUC 0,55 frente a 0,67 en los pliegues 0 y 2, en las dos configuraciones). Es la semana 12–14, donde la tasa de fraude es 3,54 % frente a 4,0–4,2 %. Refuerza que las diferencias de 0,002 entre configuraciones son ruido: la señal temporal pesa mucho más que la elección de hiperparámetros.
6. **Confirmación en `validation` (n = 77.822, fraude 3,389 %).**

| modelo | ROC-AUC | PR-AUC | precision@0,5 | recall@0,5 | F1 | positivos predichos |
|---|---:|---:|---:|---:|---:|---:|
| ganador H (config 18) | 0,9262 | 0,5814 | 0,8432 | 0,3405 | 0,4851 | 1,37 % |
| `lgbm` Fase G | 0,9259 | 0,5847 | 0,8441 | 0,3428 | 0,4876 | 1,37 % |

La confirmación **no mejora** a la referencia de Fase G: −0,0033 de PR-AUC, +0,0004 de ROC-AUC, dentro del ruido. Es exactamente lo esperable dado el punto 1, y es la razón por la que no se tomó `validation` como criterio de selección: si se hubiera elegido por `validation`, se habría favorecido la configuración de Fase G por una diferencia que el diseño no puede distinguir. Se registra el resultado negativo tal cual (`constitution.md` §6).

## 7. Modelo final

Ajustado sobre `train` completo (434.176 filas) con la configuración ganadora y `n_estimators=219` (mediana del early stopping por pliegue), y serializado en `artifacts/model_final.joblib` (3,1 MB). El artefacto contiene **solo el estimador**: la procedencia queda registrada en `selected.json` (configuración, semilla, rondas, SHA-256 de `split.parquet`), que es lo que permite reconstruirlo. Enlazar features y transformaciones al artefacto corresponde a la Fase L (pipeline de inferencia).

```python
LGBMClassifier(
    num_leaves=127,
    learning_rate=0.05,
    min_child_samples=100,
    colsample_bytree=0.6,
    subsample=1.0,
    subsample_freq=1,
    reg_lambda=0.0,
    reg_alpha=0.0,
    scale_pos_weight=1.0,
    n_estimators=219,
    deterministic=True,
    force_row_wise=True,
    n_jobs=-1,
)
```

El número de rondas se toma de la mediana de los pliegues y **no** se ajusta contra `validation` ni `test`.

## 8. Reproducibilidad (specs §15, constitution §2)

- `SEED=42` en el muestreo de configuraciones y en los parámetros de LightGBM. `sample_configs` es determinista y hay test de ello: las mismas 20 configuraciones, incluido el anclaje en la posición 0.
- `deterministic=True` y `force_row_wise=True`; verificado en la Fase G que 1, 4 y 8 hilos dan resultados idénticos.
- `selected.json` y `search.json` registran semilla, número de configuraciones, SHA-256 de `split.parquet`, número de features, criterio de selección, desempate y `test_usado: false`.
- **Verificado:** dos ejecuciones completas de `run_tuning.py` (87 min cada una) produjeron resultados idénticos. `selected.json` coincide byte a byte, y en `search.csv` **todas las columnas coinciden bit a bit salvo `seconds`**, que registra tiempo de pared y por definición varía entre ejecuciones. Los 60 ajustes reproducen los mismos `mean_pr_auc`, `std_pr_auc`, `min_pr_auc`, `mean_roc_auc`, `mean_recall` y `best_iteration`.
- El artefacto `artifacts/model_final.joblib` se regenera en cada ejecución; su round-trip de serialización está cubierto por test.
- Tests en `tests/test_tuning.py` (21): determinismo del muestreo, exclusividad del anclaje, conversión a tipos Python, reproducibilidad de `tune_one`, aislamiento del `test`, validad de la selección y del desempate, coincidencia de la métrica de LightGBM con scikit-learn, early stopping sobre la métrica de selección, reproducibilidad del ajuste final y round-trip de serialización. Suite completa del repositorio: 64 tests.

## 9. Riesgos y pendientes

- **No hay resultado sobre `test`.** Todas las cifras proceden de `train` y `validation`. Consumir el `test` es la Fase I y es irreversible.
- **La diferencia entre la ganadora y el ancla no es interpretable.** Con ±0,05 de dispersión entre pliegues y un solo `seed`, el orden de los diez primeros puestos no es estable. Afirmar que `num_leaves=127` es mejor que `63` sería sobreinterpretar. Lo defendible es que ambas rinden igual dentro de la resolución del experimento.
- **Un solo seed.** Repetir la búsqueda con 3–5 semillas reduciría el ruido de selección y daría un rango de confianza por configuración. No se hizo por coste (87 min por ejecución).
- **20 configuraciones sobre un espacio de 17.280 combinaciones.** Cobertura del 0,1 %; no es una búsqueda exhaustiva y no pretende serlo. Fijando los tres hiperparámetros sin señal (§6, punto 4) el espacio bajaría a 480 combinaciones, que sí resultaría abordable con más semillas.
- **Selección de la ronda final por mediana de pliegues** es una heurística razonable, no una técnica de bagging. Una alternativa más robusta sería promediar las predicciones de los modelos por pliegue; queda para Fases I/J si la Fase I mostrara que la varianza de la ronda importa.
- **Tensión con `constitution.md` §7 ("preferir la solución más simple"):** la ganadora **no** es más simple que el ancla (el doble de hojas, sin regularización L2 ni bagging de filas). Se sigue el criterio declarado de §3 en lugar de introducir un criterio de simplicidad a posteriori. Se deja constancia para que la decisión se revise antes de consumir el `test` en Fase I, cuando ya no será revocable.
- **Fase I:** barrido de umbrales y elección justificada del operativo sobre el `test` aislado.
- **Fase J:** la utilidad de las 17 features de la Fase E sigue sin medirse de forma directa.
- **Fase K (MLflow):** pendiente. `search.json` y `selected.json` están en JSON plano para poder importarse como runs.
