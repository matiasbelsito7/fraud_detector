# Feature engineering — Fase E

Fecha: 2026-09-25; ampliado el 2026-09-28 con T-E02 (identidad de cliente). Módulos: `src/features/features.py` (orquestador), `src/features/client.py` (agregaciones de cliente) y `src/features/columns.py` (nombres de columnas crudas compartidos); ejecución: `run_feature_engineering.py`. Salida (gitignored): `data/processed/train_features.parquet`.

Catálogo actual: **49 features propias** (20 de tiempo/frecuencia/historial + 29 de identidad de cliente), sobre un parquet de **793 columnas** (764 en la Fase E original).

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

### Identidad de cliente (T-E02, 2026-09-28)
| Feature | Representa | Cálculo | Información | Inferencia |
|---|---|---|---|---|
| `mean_C{i}_prev_client` (i=1..14) | Media previa de la columna C en el mismo `card1+addr1` (0 si no hay historial) | `groupby(card1,addr1)` acumulado: `cumsum()-x` sobre no nulos | `card1`, `addr1`, `C1..C14`, `TransactionDT` | Sí, con historial acumulado |
| `mean_C{i}_prev_uid` (i=1..14) | Ídem, restringido al mismo stint de tarjeta (`+D1n`) | ídem con clave `card1+addr1+D1n` | `card1`, `addr1`, `C1..C14`, `D1`, `TransactionDT` | Sí, con historial acumulado |
| `cnt_client_uid` | Nº de transacciones previas del mismo `card1+addr1+D1n` | `groupby(uid).cumcount()` | ídem | Sí |

`D1n = TransactionDay - D1` es **clave interna, no feature**: su información ya está en `D1` (la edad de la tarjeta *es* `D1`) y emitirlo reintroduciría el identificador de cliente que la referencia de la competencia rechaza explícitamente.

Convención de centinela: sin historial previo se emite `0.0`, nunca `NaN`, igual que `mean_amt_card1_prev`, para no romper el check `.notna()` del pipeline ni el test de inferencia de fila única.

Degradación explícita: sin columnas C no hay medias; sin `D1` no hay nivel refinado; una fila con `D1` nulo conserva sus medias primarias, porque la clave primaria no usa `D1`. Detalle completo, evidencia y veredicto en `docs/client_features_report.md`.

## 3. Reproducibilidad (specs §15)

`run_feature_engineering.py` genera las features dos veces y compara el SHA-256 de ambas (resultado `IDENTICO: True`). El resultado se combina con `train_processed.parquet` (Fase D) en `data/processed/train_features.parquet`.

## 4. Consistencia en inferencia

`compute_features` acepta una fila única (test `test_inferencia_fila_unica_consistente`), devuelve el mismo esquema y sin valores faltantes (sentinels documentados). El orden de entrada no afecta al resultado (se resta en orden temporal). `identity` es opcional; sin ella, `has_identity`/`is_mobile` toman valores consistentes.

Las features de cliente heredan esa propiedad: una fila aislada no tiene historial, así que las 29 salen en el centinela `0.0` sin `NaN`, y el test de fila única lo comprueba. La contrapartida es que **en inferencia de una sola transacción las features de cliente no aportan nada**: su valor requiere el historial acumulado del cliente. Es la misma limitación de `mean_amt_card1_prev` y `cnt_card1`, y solo se resuelve con un pipeline de inferencia que maintenga estado (Fase L).

## 5. Alcance y pendientes

- Features utilizadas: tiempo, frecuencia, historial de importes, entidades, relaciones e identidad de cliente. Su utilidad real ya se midió: las 20 primeras quedaron respaldadas por la escalera de modelos de la Fase G, y las 29 de cliente **no están certificadas** como mejora (ver `docs/client_features_report.md`).
- La definición vectorizada de las medias de cliente está atada a su definición ingenua por `test_media_vectorizada_coincide_con_expanding`, para que la optimización no pueda divergir de la semántica de "media previa".
- Pendiente para fases futuras: agregaciones sobre más entidades (email destino, pais de tarjeta vs dirección), ventanas temporales y features de `id_*` de identity. Y el identificador-como-float (Hallazgo 5 de `docs/client_features_report.md`), que sigue sin tratar.
