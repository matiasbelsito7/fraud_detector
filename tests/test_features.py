from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.client import AGG_COLS, SENTINEL, previous_mean, refined_uid
from src.features.features import KEY, compute_features, previous_expanding


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [86400, 86500, 86400, 86600, 86700],
            "TransactionAmt": [100.0, 50.0, 200.0, 30.0, 40.0],
            "card1": [111, 111, 222, 111, 111],
            "addr1": [10.0, 10.0, 20.0, np.nan, 10.0],
            "D1": [5.0, 5.0, 3.0, np.nan, 6.0],
            "C1": [1.0, 3.0, np.nan, 5.0, 7.0],
            "C2": [np.nan, np.nan, np.nan, 2.0, np.nan],
            "M1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", np.nan, np.nan],
            "R_emaildomain": ["gmail.com", "hotmail.com", "yahoo.com", np.nan, np.nan],
        }
    )


def test_frecuencias_previas() -> None:
    feats = compute_features(_toy_df())
    feats.index = _toy_df()[KEY]
    assert feats.loc[1, "cnt_card1"] == 0
    assert feats.loc[2, "cnt_card1"] == 1
    assert feats.loc[4, "cnt_card1"] == 2
    assert feats.loc[3, "cnt_card1"] == 0
    assert feats.loc[2, "cnt_card1_addr1"] == 1
    assert feats.loc[4, "cnt_card1_addr1"] == 0


def test_historial_previo_de_importes() -> None:
    feats = compute_features(_toy_df())
    feats.index = _toy_df()[KEY]
    assert feats.loc[2, "mean_amt_card1_prev"] == 100.0
    assert np.isclose(feats.loc[2, "amt_rel_card1"], 0.5)
    assert feats.loc[4, "mean_amt_card1_prev"] == 75.0
    assert feats.loc[1, "amt_rel_card1"] == 1.0


def test_gap_y_since_first_orden_por_tiempo() -> None:
    feats = compute_features(_toy_df())
    feats.index = _toy_df()[KEY]
    assert feats.loc[2, "gap_sec_card1"] == 100
    assert feats.loc[4, "gap_sec_card1"] == 100
    assert feats.loc[4, "since_first_sec_card1"] == 200
    assert feats.loc[1, "since_first_sec_card1"] == 0


def test_email_igual_remitente_destinatario() -> None:
    feats = compute_features(_toy_df())
    feats.index = _toy_df()[KEY]
    assert feats.loc[1, "p_eq_r_email"] == 1
    assert feats.loc[2, "p_eq_r_email"] == 0
    assert feats.loc[3, "p_eq_r_email"] == 1
    assert feats.loc[4, "p_eq_r_email"] == -1


def test_sin_retroactividad_ni_info_futura() -> None:
    df = _toy_df()
    base = compute_features(df)
    base.index = df[KEY]
    cambiada = df.copy()
    cambiada.loc[cambiada[KEY] == 4, "TransactionAmt"] = 9999.0
    cambiada.loc[cambiada[KEY] == 4, "addr1"] = 99.0
    alt = compute_features(cambiada)
    alt.index = cambiada[KEY]
    for col in base.columns:
        assert np.array_equal(base.loc[1:2, col], alt.loc[1:2, col]), col


def test_inferencia_fila_unica_consistente() -> None:
    df = _toy_df()
    one = compute_features(df.iloc[[0]])
    assert one.shape[0] == 1
    assert one.loc[0, "cnt_card1"] == 0
    assert one.loc[0, "mean_amt_card1_prev"] == 0.0
    assert one.loc[0, "amt_rel_card1"] == 1.0
    assert one.notna().all().all()
    cliente = [
        c for c in one.columns if c.startswith(("mean_C", "mean_M", "cnt_client"))
    ]
    assert len(cliente) == 5
    assert (one[cliente] == SENTINEL).all().all()


def test_determinismo_2_pasadas() -> None:
    df = _toy_df()
    a = compute_features(df)
    b = compute_features(df)
    pd.testing.assert_frame_equal(a, b)


def test_devices_sin_identity_consistente() -> None:
    df = _toy_df()
    feats = compute_features(df)
    feats.index = df[KEY]
    assert (feats["has_identity"] == 0).all()
    assert (feats["is_mobile"] == -1).all()


CLIENT_COLS = [
    "cnt_client_uid",
    "mean_C1_prev_client",
    "mean_C2_prev_client",
    "mean_C1_prev_uid",
    "mean_C2_prev_uid",
]


def _cliente() -> pd.DataFrame:
    feats = compute_features(_toy_df())
    feats.index = _toy_df()[KEY]
    return feats[CLIENT_COLS]


#: Columnas C presentes en `_toy_df`, para iterar sin codificar la lista a mano.
AGG_COLS_PRESENTES = [c for c in AGG_COLS if c in _toy_df().columns]


def test_clave_refinada_separa_mismo_cliente() -> None:
    """`card1+addr1+D1n` afina lo que `card1+addr1` agrupa: el plastic se reutiliza.

    ID5 comparte `card1+addr1` con ID1 e ID2, pero su `D1n` es distinto (`-5` frente
    a `-4`), asi que arranca un stint nuevo. La clave primaria lo promedia con el
    historial del stint anterior; la refinada no.
    """
    feats = _cliente()
    assert feats.loc[5, "mean_C1_prev_client"] == 2.0
    assert feats.loc[5, "mean_C1_prev_uid"] == SENTINEL
    assert feats.loc[5, "cnt_client_uid"] == 0
    assert feats.loc[2, "mean_C1_prev_client"] == 1.0
    assert feats.loc[2, "mean_C1_prev_uid"] == 1.0
    assert feats.loc[2, "cnt_client_uid"] == 1


def test_media_vectorizada_coincide_con_expanding() -> None:
    """El atajo `cumsum() - x` no puede divergir de `expanding().mean().shift(1)`.

    Es el test critico de T-E02: la implementacion vectorizada evita ~495.000
    llamadas de Python por clave, y esta comparacion la ata a su definicion para
    que la optimizacion no pueda divergir de la semantica de la media previa.
    """
    ordered = _toy_df().sort_values("TransactionDT", kind="stable")
    keys_primaria = [ordered["card1"], ordered["addr1"]]
    uid = refined_uid(ordered)
    assert uid is not None
    sets = {"client": keys_primaria, "uid": [*keys_primaria, uid]}
    for col in AGG_COLS_PRESENTES:
        for name, keys in sets.items():
            atajo = previous_mean(ordered, keys, col, f"mean_{col}_{name}").to_numpy()
            definicion = previous_expanding(
                ordered, keys, col, "mean", f"ref_{name}"
            ).to_numpy()
            sin_historial = np.isnan(definicion)
            assert (atajo[sin_historial] == SENTINEL).all(), f"{col}/{name}"
            np.testing.assert_allclose(
                atajo[~sin_historial],
                definicion[~sin_historial],
                err_msg=f"{col}/{name}",
            )


def test_columnas_m_no_se_agregan() -> None:
    """M1-M9 son categoricas en crudo y quedan fuera de la agregacion (T-E02)."""
    feats = compute_features(_toy_df())
    assert "M1" in _toy_df().columns
    assert not [c for c in feats.columns if "M1" in c]
    assert not [c for c in feats.columns if c.startswith("mean_M")]


def test_agregaciones_de_cliente_sin_info_futura() -> None:
    """Cambiar la ultima transaccion no altera las medias de cliente anteriores."""
    df = _toy_df()
    base = _cliente()
    cambiada = df.copy()
    ultima = cambiada[KEY] == df[KEY].max()
    cambiada.loc[ultima, "C1"] = 999.0
    cambiada.loc[ultima, "C2"] = 999.0
    cambiada.loc[ultima, "M1"] = 999.0
    cambiada.loc[ultima, "D1"] = 999.0
    alt = compute_features(cambiada)[CLIENT_COLS]
    alt.index = cambiada[KEY]
    for col in CLIENT_COLS:
        assert np.array_equal(base.loc[1:4, col], alt.loc[1:4, col]), col
    assert alt.loc[5, "mean_C1_prev_uid"] == SENTINEL


def test_cliente_sin_historial_usa_centinela() -> None:
    """Sin historial se emite `0.0`, nunca `NaN`, y con historial el valor es real."""
    feats = _cliente()
    assert feats.notna().all().all()
    assert feats.loc[1, "mean_C1_prev_client"] == SENTINEL
    assert feats.loc[1, "mean_C1_prev_uid"] == SENTINEL
    # ID4 no tiene `D1` ni `addr1`: degrada al nivel primario, tambien con centinela.
    assert feats.loc[4, "mean_C1_prev_client"] == SENTINEL
    assert feats.loc[4, "mean_C1_prev_uid"] == SENTINEL
    # `C2` es nula en todas las filas previas de su grupo: centinela, no NaN.
    assert feats.loc[5, "mean_C2_prev_client"] == SENTINEL
    # Y con historial real el centinela no es un valor fijo.
    assert feats.loc[5, "mean_C1_prev_client"] == 2.0


def test_degrada_sin_columnas_c_m_d1() -> None:
    df = _toy_df()

    sin_d1 = compute_features(df.drop(columns=["D1"]))
    assert "cnt_client_uid" not in sin_d1.columns
    assert not [c for c in sin_d1.columns if c.endswith("_prev_uid")]
    assert "mean_C1_prev_client" in sin_d1.columns
    assert sin_d1.notna().all().all()

    sin_c = compute_features(df.drop(columns=["C1", "C2"]))
    assert not [c for c in sin_c.columns if c.startswith(("mean_C", "mean_M"))]
    assert "cnt_client_uid" in sin_c.columns

    pelado = compute_features(df.drop(columns=["D1", "C1", "C2", "M1"]))
    assert not [c for c in pelado.columns if "client" in c or "prev_uid" in c]
    assert pelado.notna().all().all()
