# Feature engineering — Fase E

Fecha: 2026-09-25. Módulo: `src/features/features.py`; ejecución: `run_feature_engineering.py`. Salida (gitignored): `data/processed/train_features.parquet`.

## 1. Principio de no-leakage

Todas las features históricas se calculan ordenando por `TransactionDT` (estable) y usando **solo filas previas en ese orden** (`shift(1)`, `cumcount`, `expanding(...).shift(1)`). Las transacciones en el mismo segundo se consideran simultáneas (nunca se usa información de un `TransactionDT` mayor). `groupby(dropna=False)` agrupa entidades desconocidas sin descartarlas.

Verificación automatizada en `tests/test_features.py::test_sin_retroactividad_ni_info_futura`: cambiar el importe/dirección de una transacción posterior (misma tarjeta) **no altera** las features de las anteriores.

## 2. Catálogo de features

Cada feature: **representa** · **cálculo** · **información que usa** · **disponibilidad en inferencia**.

### Ciclos temporales
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `hour_sin` / `hour_cos` | Hora del día en ciclo | `(DT mod 86400)/3600`, seno/coseno | `TransactionDT` | Sí (siempre) |
| `dow_sin` / `dow_cos` | Día de semana en ciclo | `(DT/86400) mod 7`, seno/coseno | `TransactionDT` | Sí (siempre) |

### Frecuencias previas
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `cnt_card1` | Nº de transacciones previas con la misma tarjeta | `groupby(card1).cumcount()` | `card1`, `TransactionDT` | Sí |
| `cnt_addr1` | Nº de transacciones previas en la misma dirección | `groupby(addr1).cumcount()` | `addr1`, `TransactionDT` | Sí |
| `cnt_P_emaildomain` | Nº de transacciones previas con el mismo email comprador | `groupby(P_emaildomain).cumcount()` | `P_emaildomain`, `TransactionDT` | Sí |
| `cnt_card1_addr1` | Nº de transacciones previas con la misma tarjeta y dirección | `groupby(card1, addr1).cumcount()` | `card1`, `addr1`, `TransactionDT` | Sí |

### Historico de importes por tarjeta
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `mean_amt_card1_prev` | Importe medio previo de la tarjeta (0 si no hay historial) | `expanding().mean().shift(1)` sobre `TransactionAmt` por `card1` | `card1`, `TransactionAmt`, `TransactionDT` | Sí |
| `std_amt_card1_prev` | Desviación previa del importe (0 si <2 historial) | `expanding().std().shift(1)` por `card1` | ídem | Sí |
| `amt_rel_card1` | Importe de la transacción / medio previo (1 si sin historial); cuán atípica es | `TransactionAmt / mean_prev` | ídem | Sí |

### Comportamiento temporal por tarjeta
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `gap_sec_card1` | Segundos desde la última transacción de la tarjeta (-1 si no hay) | `TransactionDT - prev(TransactionDT)` por `card1` | `card1`, `TransactionDT` | Sí |
| `since_first_sec_card1` | Segundos desde la primera transacción previa de la tarjeta | `TransactionDT - min_prev(TransactionDT)` por `card1` | ídem | Sí |

### Entidades y relaciones
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `nunique_addr1_card1` | Nº de direcciones distintas vistas en la tarjeta (entre las previas); "address velocity" | `expanding().nunique()` sobre `addr1` por `card1`, previos | `card1`, `addr1`, `TransactionDT` | Sí |
| `p_eq_r_email` | 1 si email comprador == destino, 0 si difieren, -1 si se desconoce alguno | comparación `P_emaildomain == R_emaildomain` | `P/R_emaildomain` | Sí |
| `has_identity` | 1 si hay fila de identity (DeviceType), 0 si no | merge `identity.DeviceType` | `identity` | Solo si se dispone de identity (si no: 0) |
| `is_mobile` | 1 mobile, 0 desktop, -1 desconocido/sin identity | `DeviceType == "mobile"/"desktop"` | `identity` | Solo si se dispone de identity (si no: -1) |

## 3. Reproducibilidad (specs §15)

`run_feature_engineering.py` genera las features dos veces y compara el SHA-256 de ambas (resultado `IDENTICO: True`). El resultado se combina con `train_processed.parquet` (Fase D) en `data/processed/train_features.parquet`.

## 4. Consistencia en inferencia

`compute_features` acepta una fila única (test `test_inferencia_fila_unica_consistente`), devuelve el mismo esquema y sin valores faltantes (sentinels documentados). El orden de entrada no afecta al resultado (se resta en orden temporal). `identity` es opcional; sin ella, `has_identity`/`is_mobile` toman valores consistentes.

## 5. Alcance y pendientes

- Features utilizadas: tiempo, frecuencia, historial de importes, entidades y relaciones; quedan para **Fase G (modelado)** decidir transformaciones/escalado y para **Fase H** evaluar su utilidad real (importancia).
- Pendiente para fases futuras: agregaciones sobre más entidades (email destino, pais de tarjeta vs dirección), ventanas temporales y features de `id_*` de identity.