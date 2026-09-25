from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.features import KEY, compute_features


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [86400, 86500, 86400, 86600],
            "TransactionAmt": [100.0, 50.0, 200.0, 30.0],
            "card1": [111, 111, 222, 111],
            "addr1": [10.0, 10.0, 20.0, np.nan],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", np.nan],
            "R_emaildomain": ["gmail.com", "hotmail.com", "yahoo.com", np.nan],
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
