"""Nombres de columnas crudas del dataset, compartidas por los modulos de features.

Viven aparte para que `features.py` (orquestador) y `client.py` (hoja) dependan
en una sola direccion: sin este modulo, `client.py` tendria que importar de
`features.py` y `features.py` de `client.py`.
"""

from __future__ import annotations

KEY = "TransactionID"
DT = "TransactionDT"
AMT = "TransactionAmt"
CARD1 = "card1"
ADDR1 = "addr1"
D1 = "D1"
