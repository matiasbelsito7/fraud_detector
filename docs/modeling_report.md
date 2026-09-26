# Modelado — Fase G

Fecha: 2026-09-25. Módulo: `src/modeling/`; ejecución: `run_training.py`. Salidas (gitignored): `reports/modeling/comparison.csv`, `reports/modeling/comparison.json`.

## 1. Alcance

Entrenar y comparar candidatos **bajo condiciones equivalentes** (misma división temporal, mismas 762 features, misma evaluación), siguiendo la estrategia progresiva de `specs.md §7`. Esta fase **no** selecciona el modelo definitivo ni fija el threshold operativo: eso es Fase H (tuning/selección) y Fase I (evaluación final).

## 2. Contrato de uso de datos (specs §6, constitution §5)

| conjunto | filas | uso en Fase G |
|---|---:|---|
| `train` | 434.176 | Ajuste de todos los candidatos |
| `validation` | 77.822 | Comparación y métricas reportadas |
| `test` | 78.542 | **No se carga.** `load_splits` recibe `EVAL_SETS=(train, validation)` y `run_training.py` verifica con `assert` que `test` no está presente |

El test sigue reservado para el cierre de la Fase H (`docs/split_report.md` §6). Los hiperparámetros de `src/modeling/models.py` son **referencias fijas, no resultados de tuning**: no se iterated ningún valor contra `validation` más de una vez.

## 3. Matriz de modelado

Entrada: `data/processed/split.parquet` (Fase F), 590.540 × 765.

- **Features: 762.** Se excluyen `TransactionID`, `isFraud` y `split_set` (`feature_columns`); se conservan las 745 columnas del preprocessing (Fase D) y las 17 features de la Fase E.
- **`TransactionDT` se mantiene como feature.** Es passthrough deliberado del preprocessing y está disponible en inferencia (una transacción nueva trae su propio timestamp). No introduce información futura, aunque sí permite que los modelos de árbol memoricen la frontera temporal del periodo de entrenamiento; es un punto a vigilar en Fase J (importancias).
- Sin imputación ni escalado adicionales: el pipeline de la Fase D ya no deja NaN.

## 4. Escalera de modelos (specs §7)

| # | Modelo | Familia | Elección | Filas de train |
|---|---|---|---|---:|
| 1 | `dummy_prior` | baseline | `DummyClassifier(strategy="prior")`: predice la tasa de fraude sin usar features. Suelo honesto, no un modelo | 434.176 |
| 2 | `logreg` | clásico | `LogisticRegression(lbfgs, C=1)`: referencia lineal; verifica que hay señal más allá del baseline | 120.000 |
| 3 | `random_forest` | clásico | `RandomForestClassifier(300, depth=12, balanced_subsample)`: referencia de bagging con tratamiento explícito del desbalance | 120.000 |
| 4 | `lgbm` | boosting | `LGBMClassifier(600, lr=0.05, num_leaves=63, colsample=0.6)`: referencia de boosting | 434.176 |
| 5 | `lgbm_pos_weight` | boosting | Igual que 4 con `scale_pos_weight=27.47`, calculado sobre `train`. Aísla el efecto de compensar el desbalance 3,5 % / 96,5 % | 434.176 |

**Decisión de infraestructura:** se añadió **LightGBM 4.7.0** a `pyproject.toml`. Motivo concreto: `specs.md §7` exige una etapa de boosting, y sobre 434k × 762 filas el boosting es donde está la diferencia de calidad; `HistGradientBoostingClassifier` de scikit-learn sería más lento y más débil en este dataset. No se añadió `xgboost`/`catboost`/`optuna` porque no hay todavía una razón concreta para ellos (`constitution.md §7`).

**Limitación honesta de la comparación:** `logreg` y `random_forest` se entrenan sobre una submuestra determinista de 120.000 filas (28 % de `train`) para acotar el pico de memoria (762 features en float64 sobre 434k filas ≈ 2,6 GB; la máquina tiene 13,9 GB). Sus métricas **no son directamente comparables** con las de los modelos de boosting entrenados sobre todo `train`: arrastran un handicap de datos. La comparación entre sí dentro de la familia de clásicos sí es justa (misma submuestra y mismo seed).

## 5. Pliegues temporales internos de train

Pendiente declarado en `docs/split_report.md` §8, resuelto aquí por `src/modeling/folds.py`: **walk-forward de ventana expansiva**, 3 pliegues de 3 semanas dentro de `train` (18 semanas), sin randomly shuffled y sin salir de `train`.

| pliegue | n_train | n_valid | sem_train_max | sem_valid | fraude % valid |
|---:|---:|---:|---:|---|---:|
| 0 | 233.881 | 63.662 | 8 | 9–11 | 4,009 |
| 1 | 297.543 | 71.089 | 11 | 12–14 | 3,536 |
| 2 | 368.632 | 65.544 | 14 | 15–17 | 4,223 |

`check_folds` valida y `run_training.py` aborta si falla: pliegues no vacíos, training precede a validación en cada pliegue, ventanas de validación disjuntas y ventana de training expansiva. La infraestructura queda lista; **su uso para tuning es Fase H**.

## 6. Métricas (specs §8.1)

`src/modeling/metrics.py` calcula el conjunto mínimo obligatorio: ROC-AUC, PR-AUC, recall en threshold, precision, F1 y matriz de confusión (`tn`/`fp`/`fn`/`tp`).

- **Criterio de comparación: PR-AUC** (`average_precision_score`), por el desbalance 3,5 % (EDA §5, `constitution.md` §5). ROC-AUC se reporta como referencia. **Accuracy no se usa** (quedaría en ~96,5 % para cualquier modelo, incluido el baseline).
- Las métricas por threshold se reportan en **0,5 como umbral de referencia**. Elegir el umbral operativo es Fase I (`specs.md` §8.2); hacerlo aquí fijaría una decisión de negocio sin fundamento.
- PR-AUC se calcula como *average precision* (área bajo la curva de precisión-recall con interpolación por escalones), no como trapecio: es la convención para clases muy desbalanceadas y no premia los tramos de baja precisión.

## 7. Resultados (evaluación en `validation`, n = 77.822, fraude 3,389 %)

Ordenados por PR-AUC. Matriz de confusión en umbral 0,5.

| modelo | familia | ROC-AUC | PR-AUC | precision | recall | F1 | tn | fp | fn | tp | fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lgbm` | boosting | **0,9259** | **0,5847** | 0,8441 | 0,3428 | 0,4876 | 75.018 | 167 | 1.733 | 904 | 167,3 |
| `lgbm_pos_weight` | boosting | 0,9182 | 0,5604 | 0,4221 | **0,6166** | **0,5012** | 72.959 | 2.226 | 1.011 | 1.626 | 172,0 |
| `random_forest` | clásico | 0,8682 | 0,4042 | 0,1917 | 0,6443 | 0,2955 | 68.022 | 7.163 | 938 | 1.699 | 84,2 |
| `logreg` | clásico | 0,6282 | 0,0753 | 0,0000 | 0,0000 | 0,0000 | 75.185 | 0 | 2.637 | 0 | 17,6 |
| `dummy_prior` | baseline | 0,5000 | 0,0339 | 0,0000 | 0,0000 | 0,0000 | 75.185 | 0 | 2.637 | 0 | 0,1 |

Referencias: PR-AUC del base rate en `validation` = 0,0339 (`dummy_prior` la iguala exactamente, como corresponde a un modelo sin señal).

### Lectura

1. **El baseline funciona como suelo esperado.** ROC-AUC 0,5 y PR-AUC igual a la tasa base confirman que el pipeline de métricas no premia trivialmente y que la ganancia de los otros modelos es real.
2. **Resultado negativo registrado: el modelo lineal aporta casi nada.** `logreg` alcanza PR-AUC 0,0753 contra 0,0339 del baseline, y a umbral 0,5 no detecta ni un solo fraude (recall 0). Las 762 features son casi en su mayoría categóricas codificadas ordinalmente y muy colineales; una frontera lineal no las separa. No se oculta este resultado (`constitution.md` §6).
3. **El salto real está en el boosting.** `lgbm` multiplica por 17 el PR-AUC del baseline (0,585 vs 0,034) y llega a ROC-AUC 0,926, el orden de magnitud esperado para este dataset.
4. **Compensar el desbalance tiene un coste medible.** `lgbm_pos_weight` **pierde** PR-AUC (0,5604 vs 0,5847) y ROC-AUC, pero **gana** recall en 0,5 (0,617 vs 0,343) a costa de precisión (0,422 vs 0,844) y de multiplicar por 3,6 los falsos positivos. No es un modelo "mejor" o "peor": implementa otro punto de la curva. Cuál sirva como operativo depende del coste relativo de un falso positivo frente a un fraude perdido, que es Fase I. `constitution.md` §5 obliga a tratar el desbalance explícitamente, y esta comparación es exactamente ese tratamiento.
5. **El recall de `lgbm` en 0,5 (0,343) no es un defecto del modelo.** Un umbral de 0,5 en una clase positiva al 3,4 % es arbitrario; por eso el umbral operativo queda para Fase I.

## 8. Reproducibilidad (specs §15, constitution §2)

- `SEED=42` en `src/modeling/models.py`, aplicado a todos los modelos con aleatoriedad (`LogisticRegression`, `RandomForestClassifier`, `LGBMClassifier`). La submuestra clásica usa `np.random.default_rng(seed)`, independiente del estado global de NumPy.
- LightGBM corre con `deterministic=True` y `force_row_wise=True`: la construcción de histogramas no depende del hardware ni del número de hilos.
- `run_training.py` registra el **SHA-256 de `split.parquet`** (`6d612a6a…`), la semilla, el umbral y las métricas de cada modelo en `reports/modeling/comparison.json`.
- **Verificado:** dos ejecuciones completas de `run_training.py` produjeron métricas idénticas en las 6 cifras (`lgbm` ROC-AUC 0,925895 / PR-AUC 0,584723 en ambas). Los modelos no se serializan en esta fase; se reentrenan al reproducir, lo que satisface "puede entrenarse y reproducirse con resultados comparables". La persistencia del modelo seleccionado corresponde a Fase H/Fase L.
- Tests en `tests/test_modeling.py` (19): determinismo de la comparación, corrección de las métricas, exclusión de columnas de control, aislamiento del test, validad de los pliegues y deteción de información futura.

## 9. Riesgos y pendientes

- **No hay resultado sobre `test`.** Todas las cifras son de `validation`, un periodo de 4 semanas posterior al de entrenamiento. No son evidencia de rendimiento en producción (`constitution.md` §10).
- `random_forest` y `logreg` arrastran el handicap de la submuestra de 120k filas (§4); reentrenarlos sobre `train` completo podría elevarlos.
- **Fase H:** tuning sobre los pliegues walk-forward de §5 y sobre `validation`, con el test intacto. Candidatos naturales: `num_leaves`, `min_child_samples`, `colsample_bytree`, y la decisión entre `lgbm` y `lgbm_pos_weight`.
- **Fase I:** barrido de umbrales y elección justificada del operativo sobre el test aislado.
- **Fase K (MLflow):** pendiente. `specs.md` §10 lo hace obligatorio; se integra después de G para no mezclar dos tareas de `tasks.md`. Los resultados de esta fase están en JSON/CSV para poder importarse como runs.
- **Fase J:** la utilidad real de las 17 features de la Fase E y de las 745 del preprocessing todavía no está medida; solo se infiere de las importancias.
