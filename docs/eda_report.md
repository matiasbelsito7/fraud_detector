# EDA — IEEE-CIS Fraud Detection

Fecha: 2026-09-25. Resultados: `reports/eda/eda_results.json`; gráficos en `reports/eda/*.png`.

## 1. Alcance y método

- Carga de `train_transaction.csv` (590,540 x 394) y `train_identity.csv` (144,233 x 41).
- Join left por `TransactionID` para el análisis de variables de identity (cobertura 23.84%).
- Análisis de: tipos de datos, missing values, cardinalidad, distribuciones, target, temporalidad, correlaciones y tasas de fraude por categoría.

## 2. Tipos de datos

| Grupo | Cantidad | Comentario |
|---|---|---|
| Numéricas (float64) | 376 | Incluye V*, C*, D*, dist*, `TransactionAmt` |
| Numéricas (int64) | 4 | `TransactionID`, `TransactionDT`, `isFraud`, `card1` |
| String | 14 | `ProductCD`, `card4`, `card6`, `P/R_emaildomain`, `M1-M9` |

Hallazgo de tooling: pandas 3.0 lee strings como dtype `str` (no `object`); importante para el pipeline de encoding.

## 3. Missing values

- `transaction`: global 41.07% de celdas faltantes. Distribución por columnas:
  - sin missing: 20 | 0-10%: 60 | 10-50%: 108 | 50-90%: 172 | >90%: 2 (`dist2` 93.6%, `D7` 93.4%).
- Missing por familia: **D1-D15** 58.15% (prom.), **V1-V339** 43.04% (prom.), C1-C14 sin missing.
- `identity`: id_* con media 37.86% de missing; 9 columnas >90% (`id_24`, `id_25`, `id_07`, `id_08`, etc.), `DeviceInfo` 17.7%.
- Patrón: la missingness es dominante en columnas V/D y en identity id_*; no es aleatoria (bloques de columnas V correlacionadas en el mismo % de faltante).

## 4. Cardinalidad

- Categóricas reales de baja cardinalidad: `ProductCD` (5), `card4` (4), `card6` (4), `M1-M9` (2-3 valores).
- `P_emaildomain` (59) y `R_emaildomain` (60) con 15.99% y 76.75% de missing respectivamente.
- `card1`-`card3` y `addr1`/`addr2` se comportan como categóricas de alta cardinalidad aunque numéricas.

## 5. Target y desbalance

- `isFraud`: 20,663 fraudes (3.50%) vs 569,877 no fraude (96.50%).
- Desbalance fuerte y confirmado → impacta elección de métricas (PR-AUC, recall en threshold) y manejo de clases.

## 6. Temporal

- `TransactionDT` oscila entre 86,400 s y 15,811,131 s desde la referencia → **182 días** de datos.
- Tasa de fraude por semana: min 1.85%, max 5.06% (p90 4.32%) → el fraude **no es estacionario**; hay régimen variable a lo largo del tiempo.
- Tasa de fraude por hora: pico a las **7:00 (10.61%)**, valle a las 13:00 (2.29%).

## 7. Importes

- `TransactionAmt`: sin missing, sin valores <= 0. Media 135.03; máx. 31,937.
- Fraude: media 149.24 vs 134.51 en no fraude; mediana 75 vs 68.5 → los fraudes tienden a importes algo mayores pero con amplia superposición.

## 8. Correlaciones con el target

- Top correlaciones lineales moderadas y positivas, todas en columnas **V**: `V257` (0.383), `V246` (0.367), `V244` (0.364), `V242` (0.361), `V201` (0.328). Hay también correlaciones negativas relevantes (ver `png_corr.png`).
- Las columnas V top están entre las que tienen **más missing>10%** (V153, V140, V257, V246, etc.) → la missingness está asociada al poder predictivo; tratarla con indicador de faltante puede aportar.

## 9. Tasas de fraude por categorías (diferencias fuertes)

- `ProductCD`: C = 11.69% vs W = 2.04%.
- `card3` = 185: 13.07% (vs valor dominante 150: 2.46%).
- `card6`: credit = 6.68% vs debit = 2.43%.
- `addr2` = 60: 9.05%; `R_emaildomain` gmail 11.92%, outlook 16.51%.
- `M4` = M2: 11.37%.
- `DeviceType`: mobile = 10.17%, desktop = 6.52% (solo 23.84% de transacciones tiene identity).

## 10. Anomalías y notas

- Sin columnas constantes.
- Sin importes negativos ni nulos en `TransactionAmt`.
- La missingness masiva (>50% en 174 columnas) sugiere fuentes de datos opcionales: **cuidar la imputación** (mejor indicadores + columnas originales).
- No se detectó duplicidad en claves (heredado de la validación de la Fase B).

## 11. Conclusiones para preprocessing (Fase D)

- Tabajar la missingness con **indicadores de faltante** y no con imputación agresiva; evaluar descartar columnas con >90% de missing (2 en txn) salvo justificación.
- Encoding: categóricas de baja cardinalidad (ProductCD, card4, card6, M) → ordinal/one-hot; alta cardinalidad (card1-3, addr, email) → frecuencia u ordinales agregadas.
- `TransactionAmt` amerita transformación logarítmica para suavizar la cola derecha.
- Columnas V/D con mucho missing requieren decisión firme (conservar + indicador) antes de modelar.

## 12. Conclusiones para feature engineering (Fase E)

- **Temporales:** hora del día (pico 7h), día de semana, distancia al primer evento del cliente/tarjeta (gaps), velocidad entre transacciones.
- **Frecuencia y comportamiento histórico:** nº de transacciones / importe acumulado por `card1`/`addr1`/`email`/`DeviceType` en ventana pasada; importe relativo de la transacción vs. historial del cliente/tarjeta.
- **Regla crítica de no-leakage:** todo agregado histórico debe usar **solo transacciones anteriores en el tiempo** (TransactionDT menor). Verificar contra `constitution.md` §3.
- **Identity:** combinar `DeviceType`/`DeviceInfo` y email; cubre solo 24% → usar indicador de presencia y, si se agrega por cliente, solo con historial previo.
- **Relaciones:** monto vs. media por tarjeta, frecuencia de tarjeta en zonas horarias distintas, combinaciones `card`+`addr`.

## 13. Conclusiones para evaluación (Fase H-I)

- No usar accuracy como métrica primaria (desbalance 3.5%).
- Priorizar **PR-AUC**, recall en threshold operativo y matriz de confusión; ROC-AUC como referencia.
- Dada la no estacionariedad semanal, la **división temporal** (no aleatoria) y un test fuera del rango de entrenamiento son obligatorios (referencia `specs.md` §6).

## 14. Riesgos y trabajo pendiente

- No se analizó a fondo la interacción entre missingness y target (p. ej., tasa de fraude según cantidad de V nulas) → candidato a feature y a verificar en EDA avanzado.
- Las correlaciones V justificarían consistencia entre train/inferencia al tratar su missingness.
- Pendiente: definir límites exactos del split temporal según las semanas observadas (Fase F).