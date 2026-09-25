# Preprocessing — Fase D

Fecha: 2026-09-25. Pipeline: `src/preprocessing/preprocessor.py`; ejecución: `run_preprocessing.py`. Artefactos (gitignored): `artifacts/preprocessor.joblib` y `data/processed/train_processed.parquet`.

## 1. Decisiones y su fundamento en el EDA

| Decisión | Implementación | Justificación (EDA) |
|---|---|---|
| Drop `dist2` y `D7` | Lista `DROP_COLS` | `dist2` 93.6% y `D7` 93.4% de missing (`eda_results.json`), sin valor predictivo aprovechable y caen en el bucket >90% (§3 del reporte EDA). |
| Indicador de faltante en numéricas | `SimpleImputer(median, add_indicator=True)` | El 41% de las celdas de `transaction` faltan y las columnas V con mayor correlación (V257, V246, V244…) están entre las de mayor missing: el *patrón de ausencia es informativo* (EDA §3 y §8). La imputación por mediana mantiene la señal sin introducir sesgo fuerte. |
| Encoding de categóricas | `OrdinalEncoder(handle_unknown=-1, encoded_missing_value=-1)` | Baja cardinalidad (`ProductCD` 5, `card4` 4, `card6` 4, `M1-M9` ≤3) y alta (–`card1-3`, `addr1-2`) pero numéricas en origen; el encoding ordinal preserva el orden de aparición y define `-1` para categorías nuevas o ausentes (consistencia en inferencia). |
| `TransactionAmt` | `log1p` + `RobustScaler` | Cola derecha fuerte (max 31,937; EDA §7); el log la suaviza y el RobustScaler reduce la influencia de valores extremos. |
| Escalado de numéricas | `RobustScaler` | Las columnas V/D están en escalas dispares y tienen outliers; mediana+IQR es robusto (EDA §7/§8). |
| `TransactionDT` pasa tal cual | `passthrough` | Es tiempo (no imputable); las features temporales escaladas se derivan en la Fase E. |
| `TransactionID`/`isFraud` fuera del pipeline | Excluidas como identidad/target | No son features; se conservan junto al resultado para trazabilidad. |

## 2. Roles de columna

`get_roles(df)` asigna: `numeric` (numéricas restantes, imputación+escalado), `amt` (`TransactionAmt`), `categorical` (`CATEGORICAL_COLS`), `passthrough` (`TransactionDT`) y `drop` (`dist2`, `D7`). Valida que existan las columnas requeridas (`TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`).

## 3. Reproducibilidad (specs §15)

`run_preprocessing.py`:
1. Ajusta el pipeline sobre los datos.
2. Serializa el pipeline y vuelve a transformar los mismos datos con la versión cargada.
3. Compara el SHA-256 de ambas salidas; el proceso termina informando `IDENTICO: True` solo si coinciden.
- Resultado: `train_processed.parquet` (features + ID + target) y `preprocessor.joblib` (para reutilizar en inferencia).

## 4. Alcance y pendientes

- Alcance: datos de `train_transaction` (tabla principal). Las variables de identity (dispositivo, id_*) y las transformaciones de features temporales/agregadas se implementan en Fase E (feature engineering), donde se define su disponibilidad en inferencia.
- El desbalance (3.5%) y la no estacionariedad no se tratan en preprocessing; corresponden a splitting (Fase F) y modelado/evaluación (Fases G-H).