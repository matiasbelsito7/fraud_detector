# Reporte de evaluacion final (Fase I)

> **Alcance de estas cifras.** Todo lo que hay en este reporte corresponde al modelo de la **Fase H, sin features de identidad de cliente** (762 features, configuración 18, 219 rondas). La Fase I se ejecutó una sola vez y `test` **no se ha vuelto a leer** desde entonces. Con posterioridad (2026-09-28) se implementó T-E02 y se re-ejecutaron `run_tuning.py` y `run_threshold.py`; ese modelo **no tiene métricas sobre `test`** porque añadir features es selección de features y `constitution.md` §5 exige que `test` no participe de ella. Las cifras del modelo con T-E02, todas de `train` out-of-fold y `validation`, están en `docs/client_features_report.md`. Lo que se conserva aquí como estimación de desempeño en producción es la cifra del modelo sin features de cliente.

## Que se midio y con que reglas

La evaluacion final se ejecuta en `run_evaluation.py`, unico punto del pipeline
que lee `test`. Antes de tocarlo se verifica el candado de consumo y tres
precondiciones:

0. el candado `config/test_consumed.json` no registra consumos previos de `test`
   (salvo `--force` explicito, que deja traza de la reutilizacion);
1. existe `reports/evaluation/threshold.json`, escrito por `run_threshold.py`;
2. el digest de `split.parquet` sigue coincidiendo con el registrado en `threshold.json` (en la ejecución de la Fase I, `6d612a6a…`);
3. la config del umbral coincide con la config seleccionada en `run_tuning.py`.

Si cualquiera falla, el script aborta sin puntuar nada. Todas las metricas se
calculan con el umbral ya registrado: no se busca ningun parametro ni ningun
umbral aqui. La precondicion 2 era entonces una barrera real (cualquier cambio de datos la hacia fallar); tras T-E02 dejo de serlo, y es el candado (0) el que hoy impide una segunda lectura, porque no depende de los datos. Al completarse una evaluacion, el script anade un registro a `config/test_consumed.json`, de modo que la propia ejecucion arma el candado para la siguiente.

## Como se fijo el umbral (y por que no en train)

El objetivo de negocio acordado es **recall >= 0,80** (criterio mas severo que el
~0,94 de `specs.md 8`, pordecision explicita del usuario).

El primer intento calibro el umbral sobre las predicciones out-of-fold de
`train`, con los 3 pliegues walk-forward. El umbral daba recall 0,8001 en train
out-of-fold, pero **0,7763 en validation** (IC 95 % `[0,7604, 0,7922]`, que
excluye 0,80). El umbral no trasladaba, y la causa es concreta:

- los modelos out-of-fold se entrenan con 234k-369k filas; el modelo final usa
  las 434k, y con mas datos reparte los scores de forma mas apretada;
- a igual umbral, las alertas bajan de 13,5 % a 11,4 % y con ellas el recall,
  mientras la precision no se mueve (0,231 en ambos bloques);
- la tasa base tambien cae (3,39 % en validation frente a 3,91 % en train).

Un umbral calibrado sobre modelos mas debiles y aplicado al modelo mas fuerte
arrastra un sesgo sistematico a la baja. Por decision del usuario se recalibro
en `validation`, que es el **unico bloque fuera de muestra puntuado por el
artefacto exacto que se despliega**. Las predicciones out-of-fold de `train` se
conservan como corroboracion independiente.

## Cobertura out-of-fold

`walk_forward_folds` es de ventana expansiva: las primeras `n_folds *
val_weeks - 1` semanas solo se usan para entrenar y nunca se puntuan. En train
(19 semanas, 3 pliegues de 3) eso deja fuera **233.881 filas**, de modo que el
out-of-fold cubre **200.295 filas (46,1 %)**. `out_of_fold_scores` lo reporta en
vez de ocultarlo, y `run_threshold.py` aborta si la cobertura baja del 25 %.

## Resultado en test

Umbral operativo `0,029329`, fijado en validation antes de leer `test`.

| Bloque | n | ROC-AUC | PR-AUC | recall | IC 95 % | precision | F1 | alertas |
|---|---|---|---|---|---|---|---|---|
| train out-of-fold | 200.295 | 0,9239 | 0,6321 | 0,8161 | [0,8075, 0,8246] | 0,2112 | 0,3560 | 15,12 % |
| validation | 77.822 | 0,9262 | 0,5814 | 0,8002 | [0,7849, 0,8154] | 0,2107 | 0,3502 | 12,87 % |
| **test** | **78.542** | **0,9025** | **0,5282** | **0,7707** | **[0,7551, 0,7864]** | **0,1708** | **0,2796** | **15,94 %** |

Matriz de confusion en test, con el umbral operativo:

| | fraude real | no fraude real |
|---|---|---|
| **alerta** | 2.138 (TP) | 10.382 (FP) |
| **sin alerta** | 636 (FN) | 65.386 (TN) |

Barrido de thresholds, curvas ROC y PR y el detalle por pliegue quedan en
`reports/evaluation/`: `test_sweep.csv`, `test_roc_curve.csv`,
`test_pr_curve.csv`, `oof_sweep.csv`, `threshold_sweep.csv`.

## Conclusion: el objetivo de 80 % de recall no es alcanzable

**El objetivo de negocio no se cumple.** El recall en test es 0,7707 y su IC
95 % `[0,7551, 0,7864]` excluye 0,80, asi que no es ruido de muestreo: hay una
brecha real de unos 3 puntos.

La causa es deriva temporal, y se ve en la progresion de los tres bloques:

- recall: 0,8161 (train OOF) -> 0,8002 (validation) -> 0,7707 (test);
- PR-AUC: 0,6321 -> 0,5814 -> 0,5282;
- ROC-AUC: 0,9239 -> 0,9262 -> 0,9025.

El modelo **no se degrada de forma uniforme**: el ranking se mantiene alto
(ROC-AUC 0,90), pero la separacion util se estrecha. Es el patron habitual cuando
el fraudulent behavior se acerca al del cliente legitimo con el tiempo.

El margen con el que se pago el objetivo tampoco fue barato: para llegar a
0,80 en validation hubo que bajar el umbral de 0,0338 a 0,0293, lo que subio las
alertas de 11,4 % a 15,9 % y bajo la precision de 0,237 a 0,171. En test se
alertan casi **6 transacciones por cada fraude real** (15,94 % de alertas frente
a 3,53 % de fraude).

## Que sigue

`test` ya fue consumido: **no debe volver a usarse para elegir nada**. Sus
metricas son la estimacion final y no un conjunto de trabajo. Si el objetivo de
recall del 80 % es un requisito duro, las vias son de negocio, no de ajuste:

1. revisar el umbral con el equipo de riesgo y aceptar el punto operativo real
   (recall ~0,77, precision ~0,17, alertas ~16 %), o
2. abrir una fase de re-entrenamiento con datos mas recientes, features que
   capturen el drift y revision del criterio de coste.

### Consecuencia de T-E02 (2026-09-28)

T-E02 se implemento **despues** de esta evaluacion, precisamente porque este reporte es el que motivo el trabajo. Sus 29 features de identidad de cliente se certificaron solo con `train` out-of-fold y `validation`, y el veredicto fue negativo: la comparacion aislada (mismos hiperparametros, solo cambia el set de columnas) da **-0,0016 de PR-AUC medio, con los tres pliegues de distinto signo**, dentro del ruido de +-0,05. El modelo de T-E02 si mejora el punto operativo de `validation` (mismo recall con 11,30 % de alertas frente a 12,87 %), pero eso no se puede trasladar a `test` sin volver a consumirlo, y no se hara.

Queda por decidir cual de los dos modelos es el desplegable: `artifacts/model_final.joblib` (T-E02, 12,5 MB) o `artifacts/model_final_fase_h.joblib` (Fase H, 3,1 MB, el que tiene las cifras de `test`). El razonamiento y las opciones estan en `docs/client_features_report.md`. Archivos de la Fase I preservados en `reports/evaluation/fase_i_baseline/`.

## Reproduccion

```bash
uv run run_tuning.py      # Fase H: selecciona config y fija rondas
uv run run_threshold.py   # fija el umbral; NO carga test
uv run run_evaluation.py  # unico consumo de test
```

Esas tres ordenes son las de la Fase I. `run_evaluation.py` no debe volver a ejecutarse. La comprobacion de digest **dejo de proteger** con T-E02, porque `run_threshold.py` se re-ejecuto y regenero `threshold.json` con el digest vigente (`162c566a...`); por eso se anadio el candado `config/test_consumed.json`, que registra cada consumo con independencia de los datos y hace abortar al script si `test` ya figura consumido. La Fase I quedo registrada en `reports/evaluation/fase_i_baseline/test_metrics.json` y como primer consumo del candado. Volver a leer `test` exige `uv run run_evaluation.py --force`, que deja traza en el propio candado; sigue vigente la regla de `constitution.md` 5 y de `tasks.md`.
