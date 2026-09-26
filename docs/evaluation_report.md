# Reporte de evaluacion final (Fase I)

## Que se midio y con que reglas

La evaluacion final se ejecuta en `run_evaluation.py`, unico punto del pipeline
que lee `test`. Antes de tocarlo se verifican tres precondiciones:

1. existe `reports/evaluation/threshold.json`, escrito por `run_threshold.py`;
2. el digest de `split.parquet` sigue siendo `6d612a6a...` (el de la Fase H);
3. la config del umbral coincide con la config seleccionada en `run_tuning.py`.

Si cualquiera falla, el script aborta sin puntuar nada. Todas las metricas se
calculan con el umbral ya registrado: no se busca ningun parametro ni ningun
umbral aqui.

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

## Reproduccion

```bash
uv run run_tuning.py      # Fase H: selecciona config y fija rondas
uv run run_threshold.py   # fija el umbral; NO carga test
uv run run_evaluation.py  # unico consumo de test
```
