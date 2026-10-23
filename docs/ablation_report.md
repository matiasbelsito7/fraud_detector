# Reporte de robustez y ablación de encoding categórico (E0/E1)

Fecha: 2026-10-20. Alcance: comparar el encoding ordinal histórico contra el encoding categórico nativo de LightGBM con la configuración ganadora de T-E02. **No se cargan `validation` ni `test`**: la decisión se toma íntegramente dentro de `train`, como exige `constitution.md` §5.

## 1. Motivación

T-E02 dejó dos hallazgos abiertos:

1. La mejora de las features de cliente no se certificó porque no fue consistente en los tres pliegues.
2. Las columnas que el preprocessor ya declara categóricas (`ProductCD`, `card4`, `card6`, `M1`–`M9`, `P_emaildomain`, `R_emaildomain`, `card1`, `card2`, `card3`, `addr1`, `addr2`) se codifican con `OrdinalEncoder` y LightGBM las recibe como números. Es decir, el modelo corta por umbrales sobre códigos que no tienen orden.

Este reporte mide el segundo punto de forma aislada: mismos datos, mismas features, mismos pliegues, misma configuración y mismas rondas; solo cambia cómo se declaran las 19 columnas categóricas al modelo.

## 2. Protocolo

| Elemento | Decisión |
|---|---|
| Datos | Solo `train` (`n = 434.176`, 791 features, fraude 3,513 %) |
| Pliegues | 4 pliegues walk-forward de 3 semanas de validación, ventana de training expansiva |
| Cobertura OOF | 262.052 filas (60,36 % de `train`); las primeras semanas solo entrenan |
| Configuración | Ganadora de T-E02 (`selected.json`, config 11): `num_leaves=255`, `lr=0,1`, `min_child_samples=20`, `colsample=1,0`, `subsample=0,8`, `reg_lambda=10`, `scale_pos_weight=1` |
| Rondas | 439 fijas para ambos encodings, sin early stopping, para que la comparación no dependa de una ronda distinta por pliegue |
| Seeds | 42 y 43; cada seed reentrena todos los pliegues |
| Encoding A | `ordinal`: matriz tal como sale del preprocessor (`-1` = missing/desconocido) |
| Encoding B | `native`: los mismos códigos desplazados `-1 → 0` y `0..N → 1..N+1`, declarados con `categorical_feature` a LightGBM |
| Métricas primarias | PR-AUC, ROC-AUC y recall a presupuestos fijos de alerta 10 %, 12 % y 15 % |
| Incertidumbre | Bootstrap percentil (200 remuestras) por seed para PR-AUC y recall@12 % |

El desplazamiento de códigos es necesario porque LightGBM exige enteros no negativos para una columna categórica nativa. No cambia la agrupación de categorías: solo traslada las etiquetas. El parquet y el preprocessor siguen guardando el encoding ordinal histórico; el desplazamiento se aplica en la frontera del modelo, tanto en entrenamiento como en inferencia.

Los presupuestos de alerta fijan el umbral ex post sobre los scores OOF, de modo que la comparación no depende de un umbral puntual ni de la calibración. `predict_rate` puede superar levemente el objetivo por empates en el corte.

## 3. Resultados

### 3.1 Agregado (media de 4 pliegues × 2 seeds)

| Métrica | Ordinal | Nativo | Δ (nativo − ordinal) |
|---|---:|---:|---:|
| PR-AUC | 0,62760 | 0,63437 | **+0,00677** |
| ROC-AUC | 0,91472 | 0,90535 | **−0,00937** |
| Recall @ 10 % alertas | 0,73487 | 0,72468 | −0,01020 |
| Recall @ 12 % alertas | 0,76249 | 0,74732 | −0,01517 |
| Recall @ 15 % alertas | 0,79911 | 0,77575 | −0,02336 |
| Precision @ 10 % alertas | 0,29293 | 0,28887 | −0,00406 |
| Precision @ 12 % alertas | 0,25328 | 0,24824 | −0,00504 |
| Precision @ 15 % alertas | 0,21236 | 0,20615 | −0,00621 |
| Umbral @ 12 % alertas | 0,007621 | 0,003515 | −0,004106 |

La variabilidad entre seeds es mínima (PR-AUC: desvío 0,00024 ordinal y 0,00006 nativo), muy inferior a las diferencias observadas. La variabilidad entre pliegues es la relevante y se reporta abajo.

### 3.2 PR-AUC y ROC-AUC por pliegue

Δ = nativo − ordinal; entre paréntesis, los dos seeds.

| Pliegue | Δ PR-AUC seed 42 | Δ PR-AUC seed 43 | Δ ROC-AUC seed 42 | Δ ROC-AUC seed 43 |
|---|---:|---:|---:|---:|
| 0 (semanas 6–8) | +0,0210 | +0,0140 | −0,0061 | −0,0113 |
| 1 (semanas 9–11) | +0,0158 | +0,0186 | −0,0093 | −0,0088 |
| 2 (semanas 12–14) | +0,0048 | +0,0144 | −0,0123 | −0,0078 |
| 3 (semanas 15–17) | **−0,0140** | **−0,0152** | −0,0137 | −0,0130 |

- **PR-AUC mejora en 3 de 4 pliegues**, pero **empeora en el pliegue más reciente** en ambos seeds. No es una mejora consistente.
- **ROC-AUC empeora en los 8 pares pliegue–seed**, sin excepción. La pérdida media (−0,0094) es unas 20 veces el desvío entre seeds.

### 3.3 Intervalos bootstrap (por seed)

| Métrica | Ordinal seed 42 | Ordinal seed 43 | Nativo seed 42 | Nativo seed 43 |
|---|---:|---:|---:|---:|
| PR-AUC [IC 95 %] | [0,6191; 0,6374] | [0,6188; 0,6363] | [0,6257; 0,6446] | [0,6257; 0,6434] |
| Recall @ 12 % [IC 95 %] | [0,7546; 0,7710] | [0,7536; 0,7715] | [0,7386; 0,7545] | [0,7402; 0,7570] |

Los intervalos de PR-AUC se solapan ampliamente: la ganancia de +0,0068 no se distingue del ruido de muestreo. Los de recall@12 % apenas se tocan: la pérdida operativa es distinguible y va en la dirección contraria.

## 4. Lectura

El resultado no es contradictorio, pero sí incómodo: **el encoding nativo mejora el ranking global de precisión (PR-AUC) y empeora el punto operativo y el ranking global de ROC-AUC**.

- PR-AUC es el área bajo la curva precisión–recall ponderada por umbrales. El nativo gana sobre todo en la zona de muy pocas alertas (precisión alta), lo que sube el área.
- A los presupuestos que importan al negocio (10–15 % de alertas), el ordinal recupera más fraude y con algo más de precisión. El nativo exige umbrales mucho más bajos para marcar la misma proporción de alertas, señal de una distribución de scores distinta.
- ROC-AUC castiga el intercambio global entre positivos y negativos y cae de forma consistente en todos los pliegues. El ordinal conserva mejor la separación global.

Para el objetivo declarado (recall ≥ 0,80), el ordinal es preferible: a 15 % de alertas alcanza 0,7991 y el nativo 0,7758; el nativo necesitaría un presupuesto de alertas mayor para acercarse. En `validation`, el modelo desplegado alcanzó 0,8002 con 11,30 % de alertas, pero esa cifra **no es comparable** con estas: aquí se mide OOF sobre pliegues de `train`, no sobre el periodo de `validation`. La comparación válida es ordinal vs nativo dentro del mismo protocolo.

## 5. Veredicto

**El encoding categórico nativo no se certifica como mejora y no se despliega.** Motivos:

1. La ganancia de PR-AUC no es consistente: se invierte en el pliegue más reciente, en ambos seeds.
2. ROC-AUC empeora en todos los pliegues y seeds.
3. El recall a los tres presupuestos operativos empeora entre 1,0 y 2,3 puntos, con intervalos que no se solapan con los del ordinal a 12 %.
4. El proyecto selecciona por PR-AUC, pero el criterio de aceptación de negocio es recall; un modelo que sube la primera y baja el segundo no es una mejora.

Se conserva `ordinal` como encoding desplegado. El soporte `native` queda implementado y testeado en `src/preprocessing/categoricals.py`, `src/modeling/` y `src/inference/pipeline.py`, disponible vía `categorical_encoding` en `deployment.json`, para que una futura decisión no exija reescribir el pipeline. `config/deployment.json` declara explícitamente `ordinal`.

## 6. Reproducibilidad

```
uv run run_ablation.py
```

- Entrada: `data/processed/split.parquet` (SHA-256 `162c566a…`) y `reports/tuning/selected.json`.
- Salida: `reports/ablation/summary.json` (gitignored).
- `test_usado: false`; `entrenado_y_evaluado_en: train`.
- Dos ejecuciones completas produjeron los mismos agregados y los mismos `pr_auc`/`roc_auc` por pliegue.
- El script no forma parte de la cadena batch de `README.md`: es un análisis puntual, no una etapa del producto.

## 7. Límites

- **Una sola configuración.** Se usó la ganadora de T-E02 sin re-tunear para el encoding nativo. Un tuning específico podría cambiar el veredicto, pero sería un experimento nuevo y consumiría presupuesto de cómputo; además, volver a tunear sobre `train` es admisible, sobre `test` no.
- **Cuatro pliegues de 3 semanas y dos seeds.** Es más robusto que la comparación de T-E02 (3 pliegues, 1 seed), pero no agota la incertidumbre temporal.
- **Bootstrap i.i.d.** Ignora la autocorrelación temporal; sus intervalos son una aproximación, no una garantía. Se reportan junto a la variabilidad por pliegue, no en su lugar.
- **Solo las 19 columnas de `CATEGORICAL_COLS`.** `D1` no entra porque el preprocessor no la declara categórica y tratarla como tal es otra decisión de preprocessing.
- **Sin `validation` ni `test`.** Es deliberado: la ablación compara variantes, no fija el punto operativo. Las cifras de `test` del modelo desplegado siguen siendo las de Fase H; T-E02 y esta ablación no las reabren.
