"""Agrupa las 791 features del modelo en familias legibles.

Las columnas del split son un bloque plano de cientos de variables `V` y `C`
mas decenas de indicadores de missing. Leer la importancia columna a columna
no dice nada sobre que bloque del dataset mueve la prediccion; estas familias
permiten agregar gain y SHAP sin perder de vista de donde viene cada senal.

La clasificacion es determinista y no depende del modelo: solo del nombre.
"""

from __future__ import annotations

import re

FAMILIES = (
    "V (Vesta)",
    "C (contadores)",
    "D (tiempo)",
    "M (match)",
    "card (tarjeta)",
    "addr (direccion)",
    "dist (distancia)",
    "email (dominio)",
    "id_ (identity)",
    "indicador de missing",
    "FE: tiempo",
    "FE: frecuencia",
    "FE: importes",
    "FE: historial tarjeta",
    "FE: entidades",
    "FE: identidad de cliente",
    "otras",
)

_V = re.compile(r"^V\d+$")
_C = re.compile(r"^C\d+$")
_D = re.compile(r"^D\d+$")
_M = re.compile(r"^M\d+$")
_MISSING = re.compile(r"^missingindicator_")
_CLIENT = re.compile(r"^mean_C\d+_prev_(client|uid)$")

_CYCLIC = frozenset({"hour_sin", "hour_cos", "dow_sin", "dow_cos"})
_COUNTS = frozenset({"cnt_card1", "cnt_addr1", "cnt_P_emaildomain", "cnt_card1_addr1"})
_AMOUNTS = frozenset({"mean_amt_card1_prev", "std_amt_card1_prev", "amt_rel_card1"})
_CARD_HISTORY = frozenset({"gap_sec_card1", "since_first_sec_card1"})
_ENTITIES = frozenset(
    {"nunique_addr1_card1", "p_eq_r_email", "has_identity", "is_mobile"}
)


def feature_family(name: str) -> str:
    """Familia de una feature a partir de su nombre, sin consultar el modelo."""
    if name in _CYCLIC:
        return "FE: tiempo"
    if name in _COUNTS:
        return "FE: frecuencia"
    if name in _AMOUNTS:
        return "FE: importes"
    if name in _CARD_HISTORY:
        return "FE: historial tarjeta"
    if name in _ENTITIES:
        return "FE: entidades"
    if name == "cnt_client_uid" or _CLIENT.match(name):
        return "FE: identidad de cliente"
    if _MISSING.match(name):
        return "indicador de missing"
    if name.startswith("id_"):
        return "id_ (identity)"
    if name.startswith("card"):
        return "card (tarjeta)"
    if name.startswith("addr"):
        return "addr (direccion)"
    if name.startswith("dist"):
        return "dist (distancia)"
    if "emaildomain" in name:
        return "email (dominio)"
    if _V.match(name):
        return "V (Vesta)"
    if _C.match(name):
        return "C (contadores)"
    if _D.match(name):
        return "D (tiempo)"
    if _M.match(name):
        return "M (match)"
    return "otras"
