# Explainability — Fase J

Fecha: 2026-10-18. Módulos: `src/explainability/families.py`, `src/explainability/importance.py`; ejecución: `run_explainability.py`. Salidas (gitignored): `reports/explainability/importance_gain.csv`, `importance_gain_by_family.csv`, `shap_global.csv`, `shap_by_family.csv`, `summary.json`.

## 1. Alcance

Analizar **por qué** el modelo realiza sus predicciones (`specs.md` §9): importancia global, SHAP global y explicaciones individuales, asociadas al artefacto seleccionado. Esta fase es **descriptiva**: no elige features, no cambia hiperparámetros, no fija umbrales y no modifica el modelo. Por eso no participa de la selección que `constitution.md` §5 reserva a `train`/`validation`.

## 2. Contrato de uso de datos

| conjunto | uso en Fase J |
|---|---|
| `train` | **No se carga.** El booster ya contiene la importancia de sus árboles; SHAP no necesita reentrenar. |
| `validation` | **Conjunto explicado.** 77.822 filas, fraude 3,389 %. Es el único bloque fuera de muestra disponible y no se usa para decidir nada. |
| `test` | **No se carga.** `load_splits` recibe `EVAL_SETS=(train, validation)` y `run_explainability.py` verifica con `assert` que `test` no está. |

El modelo explicado es `artifacts/model_final.joblib`: LightGBM, configuración 11 (T-E02), 439 rondas y 791 features. Es el artefacto seleccionado por `run_tuning.py`, no una elección nueva. Sigue abierta la decisión entre este modelo y `artifacts/model_final_fase_h.joblib` (ver `docs/client_features_report.md`); esta fase **no la resuelve**, solo documenta el modelo que hoy ocupa el lugar de final.

## 3. Método

| lectura | técnica | qué mide |
|---|---|---|
| Importancia global | `booster.feature_importance(importance_type="gain"/"split")` | Ganancia acumulada de las divisiones construidas durante el entrenamiento y número de divisiones por feature. |
| SHAP global | media de `|SHAP|` por feature | Contribución media de cada feature a la predicción, en log-odds, con signo. |
| SHAP individual | `predict(X, pred_contrib=True)` sobre una fila | Contribución de cada feature a esa transacción concreta. |

**Decisión de implementación:** se usa el **TreeSHAP nativo de LightGBM** (`pred_contrib=True`), que devuelve valores SHAP exactos para modelos de árbol, en lugar de añadir el paquete `shap`. Motivo: `specs.md` §9 admite "SHAP u otra técnica equivalente", los valores son los mismos y la dependencia no aportaría nada (`constitution.md` §7). La última columna que devuelve LightGBM es el valor base (bias) del modelo.

**Validación de la técnica.** La invariante `bias + suma(contribuciones) == raw_score` se comprueba sobre la muestra: error máximo **2,25 × 10⁻⁸**. Si LightGBM cambiara la convención, el runner aborta.

**Muestreo.** SHAP se calcula sobre **5.000 filas de `validation`** (muestreo uniforme sin reemplazo, `seed=42`, índices ordenados). El coste de `pred_contrib` es aproximadamente lineal y muy superior al de `predict`: 5.000 filas tardan ~4 minutos con 439 árboles, por lo que ampliar la muestra encarece la fase sin cambiar las conclusiones. La muestra es determinista y se registra en `summary.json`.

**Familias.** Las 791 features se agrupan por nombre (`src/explainability/families.py`) en V, C, D, M, card, addr, dist, email, id_, indicadores de missing, features de ingeniería y "otras". La agrupación depende solo del nombre, no del modelo, y permite leer el bloque del dataset que mueve la predicción.

## 4. Importancia global

Top-10 por ganancia (sobre todos los árboles) y por `|SHAP|` (sobre la muestra de `validation`):

| # | gain (feature) | gain % | # | SHAP (feature) | \|SHAP\| medio |
|---:|---|---:|---:|---|---:|
| 1 | `V258` | 10,25 | 1 | `C13` | 0,3003 |
| 2 | `C1` | 3,81 | 2 | `since_first_sec_card1` | 0,2164 |
| 3 | `C14` | 3,38 | 3 | `TransactionAmt` | 0,2095 |
| 4 | `card1` | 2,67 | 4 | `C1` | 0,1790 |
| 5 | `cnt_P_emaildomain` | 2,45 | 5 | `C14` | 0,1592 |
| 6 | `TransactionAmt` | 2,41 | 6 | `V70` | 0,1542 |
| 7 | `C13` | 2,29 | 7 | `card6` | 0,1319 |
| 8 | `card2` | 2,28 | 8 | `cnt_P_emaildomain` | 0,1288 |
| 9 | `V294` | 2,04 | 9 | `card1` | 0,1190 |
| 10 | `cnt_client_uid` | 2,01 | 10 | `cnt_card1` | 0,1185 |

Las dos lecturas coinciden en lo esencial pero no en el orden:

- **La señal está concentrada.** El top-10 acumula 33,56 % de la ganancia y 23,0 % del `|SHAP|` total. No hay una única feature que explique el modelo.
- **`V258` es la feature de mayor ganancia (10,25 %) pero la 45.ª por `|SHAP|`** (0,0563). No es una contradicción: la ganancia acumula la mejora de las divisiones durante el entrenamiento, mientras que `|SHAP|` mide la contribución media por predicción en `validation`. Una feature puede producir pocas divisiones muy decisivas y, aun así, moverse poco en promedio.
- **Las features de ingeniería aparecen arriba.** `since_first_sec_card1` es la 2.ª por SHAP, `cnt_P_emaildomain` la 8.ª y `cnt_card1` la 10.ª; `cnt_client_uid` es la 10.ª por ganancia y la 11.ª por SHAP. La ingeniería de la Fase E no es decorativa.
- **`TransactionDT` se usa** (12.ª por ganancia, 24.ª por SHAP). Es passthrough y está disponible en inferencia, pero también permite memorizar la frontera temporal del periodo de entrenamiento; es el punto que ya advertía `docs/modeling_report.md` §3. La explicabilidad lo confirma como feature activa, no resuelve si conviene mantenerla.
- **355 features no se usan nunca** (`split=0` y `gain=0`): 333 de los 355 indicadores de missing y 22 variables V. Los indicadores de missing existen porque `SimpleImputer(add_indicator=True)` los generó para todas las columnas con nulos, pero el modelo descarta la gran mayoría.

## 5. Importancia por familia

| familia | n | gain % | `|SHAP|` acumulado | SHAP % |
|---|---:|---:|---:|---:|
| V (Vesta) | 339 | 27,72 | 1,6270 | 21,80 |
| FE: identidad de cliente | 29 | 13,28 | 1,1557 | 15,48 |
| C (contadores) | 14 | 15,68 | 1,0777 | 14,44 |
| D (tiempo) | 14 | 7,58 | 0,6307 | 8,45 |
| card (tarjeta) | 6 | 7,11 | 0,4689 | 6,28 |
| M (match) | 9 | 1,72 | 0,4545 | 6,09 |
| FE: frecuencia | 4 | 7,11 | 0,4235 | 5,67 |
| otras | 3 | 4,61 | 0,3370 | 4,52 |
| FE: importes | 3 | 4,77 | 0,2937 | 3,93 |
| FE: historial tarjeta | 2 | 2,04 | 0,2798 | 3,75 |
| FE: tiempo | 4 | 1,94 | 0,2024 | 2,71 |
| FE: entidades | 4 | 1,74 | 0,1268 | 1,70 |
| indicador de missing | 355 | 0,49 | 0,1197 | 1,60 |
| email (dominio) | 2 | 1,65 | 0,1185 | 1,59 |
| addr (direccion) | 2 | 1,77 | 0,0854 | 1,14 |
| dist (distancia) | 1 | 0,78 | 0,0628 | 0,84 |

Lectura por densidad, no solo por total: el bloque V suma mucho porque tiene 339 columnas (0,08 % de ganancia por feature), mientras que las 29 features de cliente aportan 13,28 % de ganancia (0,46 por feature) y las 14 C aportan 15,68 % (1,12 por feature). **Las 29 features de identidad de cliente son, en proporción, el segundo bloque más influyente después de C**, y `cnt_client_uid` entra en el top-10 de ganancia. Esto contradice la idea de que T-E02 sea peso muerto, pero **no la certifica como mejora**: la comparación pre-registrada por pliegue fue negativa y esa decisión no se reabre con evidencia descriptiva (`constitution.md` §5 y §6).

## 6. Explicaciones individuales

En el umbral operativo ya registrado (`0,008480`), se explica el caso de mayor score de cada grupo. Los valores SHAP están en log-odds; el valor base del modelo es **−6,9333**.

| caso | fila en `split.parquet` | score | top contribuciones (SHAP) |
|---|---|---:|---|
| Verdadero positivo | 455116 | 0,99994 | `V258` +1,974 · `C1` +1,791 · `C11` +0,820 · `V45` +0,646 · `C10` +0,531 |
| Falso negativo | 450500 | 0,00843 | `cnt_addr1` +0,455 · `TransactionAmt` +0,339 · `card6` +0,331 · `std_amt_card1_prev` +0,317 · `P_emaildomain` −0,310 |
| Falso positivo | 473709 | 0,99875 | `V258` +2,107 · `C1` +1,552 · `C13` +1,167 · `C11` +0,781 · `V187` +0,773 |

1. **El positivo y el falso positivo comparten la misma firma de alto riesgo** (`V258`, `C1`, `C11` empujan fuerte hacia fraude). El falso positivo es una transacción legítima con el patrón de las fraudulentas: el límite de precisión es estructural, no un artefacto del umbral. Es coherente con la precisión 0,1708 medida en `test` a recall 0,7707 (`docs/evaluation_report.md`).
2. **El falso negativo está a 5 × 10⁻⁵ del umbral** (0,008429 frente a 0,008480). Es el fraude mejor puntuado que se escapa: el modelo lo ordena casi como positivo, pero el corte operativo lo deja fuera. Recuerda que el umbral se fijó para el recall agregado, no garantiza cada caso.
3. **El falso negativo sí usa features de historial y frecuencia** (`cnt_addr1`, `std_amt_card1_prev`, `since_first_sec_card1`), con `P_emaildomain` en contra. La decisión no es arbitraria, solo insuficiente.
4. Los casos son deterministas y extremos (mayor score de cada grupo), elegidos para que la explicación sea legible; **no son una muestra representativa** de todos los aciertos y errores.

## 7. Reproducibilidad

- **Verificado:** tres ejecuciones completas de `run_explainability.py` (la última con el código final, que carga solo `validation`) produjeron los cinco artefactos con **SHA-256 idéntico**.
- El muestreo es determinista (`seed=42`, índices ordenados) y LightGBM corre con `deterministic=True` y `force_row_wise=True`, las mismas condiciones que el modelo.
- La aditividad de SHAP se comprueba en cada ejecución (2,25 × 10⁻⁸) y el runner aborta si supera `1e-4`.
- `summary.json` registra técnica, muestra, semilla, digest de `split.parquet`, umbral, error de aditividad, top features, agregados por familia y casos individuales. `test_usado: false`.
- Tests en `tests/test_explainability.py` (14): clasificación de familias, orden y reparto de gain, rechazo de objetos sin booster, aditividad SHAP, formas y errores de `shap_contributions`, media `|SHAP|`, contribuciones locales con signo, agregación por familia, determinismo del muestreo y aislamiento de `test` en el runner. Suite completa del repositorio: **105 tests**.

## 8. Riesgos y pendientes

- **Solo se explica `model_final.joblib`.** La decisión entre T-E02 y Fase H sigue abierta; explicar también `model_final_fase_h.joblib` permitiría comparar firmas, pero no resolvería cuál desplegar. Queda para cuando se decida el modelo.
- **Muestra de 5.000 filas.** Es suficiente para el ranking global, pero no se reportan intervalos de confianza por feature; diferencias pequeñas de `|SHAP|` entre puestos contiguos no son interpretables.
- **Los casos individuales son extremos**, no representativos. La explicación sirve para entender el patrón, no para inferir tasas de acierto.
- **SHAP es asociación, no causalidad.** Una contribución alta indica que la feature movió la predicción, no que sea la causa del fraude.
- **Hallazgo para una fase futura:** 333 de los 355 indicadores de missing nunca se usan. Eliminarlos reduciría el ancho del modelo sin cambiar su comportamiento, pero tocar el preprocessing es otra decisión y no se hace aquí (`constitution.md` §7).
- **Fase K (MLflow):** pendiente. Estos artefactos están en CSV/JSON para poder registrarse como artefactos del run.
