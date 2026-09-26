# División de datos — Fase F

Fecha: 2026-09-25. Módulo: `src/split/split.py`; ejecución: `run_split.py`. Salida (gitignored): `data/processed/split.parquet`.

## 1. Criterio: split temporal por semanas

La estrategia es **temporal y aislada**, la única válida para transacciones (specs §6, EDA: sin estacionariedad, tasa de fraude por semana variable). `TransactionDT` es estrictamente creciente, así que ordenar por fecha e impedir que el modelo vea el futuro es directo: training = conjunto más antiguo; test = conjunto más reciente; validation = intermedia.

- Total de datos: 26 semanas (182 días), desde `TransactionDT=86400` hasta `15811131`.
- Umbrales por semana completa (7 días), deterministas y **sin semilla aleatoria**:
  - `train` = semanas 0..17 (18 semanas)
  - `validation` = semanas 18..21 (4 semanas)
  - `test` = semanas 22..25 (4 semanas)
- Configuración en constantes de `src/split/split.py`: `TRAIN_WEEKS=18`, `VAL_WEEKS=4`, `WEEK_SECONDS=7*24*3600`.

## 2. Cómo se generan los tres conjuntos

`assign_sets(TransactionDT)` asigna `week = (dt - min_dt) // WEEK_SECONDS` y etiqueta cada fila:

```
week < TRAIN_WEEKS                              -> train
TRAIN_WEEKS <= week < TRAIN_WEEKS + VAL_WEEKS   -> validation
resto                                           -> test
```

`apply_split(df)` devuelve la matriz + columna `split_set` (disjunta, completa, cubre el 100% de filas). La división **no depende del orden de las filas**: transacciones del mismo `TransactionDT` caen siempre en el mismo conjunto.

## 3. Resultados observados

| set | n | % | fraude | fraude % | dt_min | dt_max | semanas |
|---|---|---|---|---:|---:|---:|---|
| train | 434,176 | 73.52 | 15,252 | 3.513 | 86,400 | 10,972,793 | 0–17 |
| validation | 77,822 | 13.18 | 2,637 | 3.389 | 10,972,801 | 13,391,998 | 18–21 |
| test | 78,542 | 13.30 | 2,774 | 3.532 | 13,392,056 | 15,811,131 | 22–25 |

Las tasas de fraude (3.4–3.5%) son homogéneas entre conjuntos, lo que permite comparar métricas entre ellos sin confundir cambio de tasa con calidad del modelo.

## 4. Validaciones (automáticas en `check_summary` + tests)

- División completa: todas las filas etiquetadas con uno de los tres valores.
- Conjuntos no vacíos y con ambas clases (`isFraud` 0 y 1) presentes.
- Orden temporal estricto: `max sem(tra) < min sem(val) <= max sem(val) < min sem(tes)`.
- Reproducibilidad: sin azar, mismo input → misma salida (test idéntico en dos pasadas).

## 5. En qué momento se usa cada conjunto

- **train**: ajuste de modelos (Fase G) y validación cruzada temporal **interna** solo dentro de train.
- **validation**: selección de hiperparámetros y decisiones de diseño (Fase G/H) y calibración del umbral operativo (Fase I), antes del test.
- **test**: evaluación final única (Fase I). Aislado: nunca se usa para decidir.

## 6. Aislamiento del test

- El test queda apartado desde esta fase y se consume una sola vez en `run_evaluation.py` (Fase I), tras verificar que el umbral ya está registrado en `reports/evaluation/threshold.json`.
- Las features ya se construyeron sin información futura (Fase E), de modo que ninguna feature de una fila de test usa filas de validation/train posteriores en el tiempo; y ninguna fila de train/validation usa filas de test (las preceden siempre).
- `split.parquet` es el registro canónico `TransactionID -> split_set`, reproducible.

## 7. Reproducibilidad (specs §15)

El split es puramente funcional sobre `TransactionDT` (sin `random_state`), por lo que es determinista por construcción. `run_split.py` valida (`IDENTICO` de diseño) y escribe `data/processed/split.parquet`.

## 8. Alcance y pendientes

- Pendiente para Fase G: dividir internamente `train` en pliegues temporales (walk-forward) para selección de hiperparámetros; el `test` permanece intacto.
- La fila nueva en inferencia se etiqueta con las mismas reglas, pero para scoreo no requiere conjunto: usa el modelo ya entrenado.