import json
import os

import numpy as np
import pandas as pd

# ── Load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH) as f:
    _config = json.load(f)

_TARGET_COL = _config["data"]["target_col"]
_DATE_COL   = _config["data"]["date_col"]
_SKU_COL    = _config["data"]["sku_col"]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values([_SKU_COL, _DATE_COL])

    # Calendar features
    df["month"]   = df[_DATE_COL].dt.month
    df["quarter"] = df[_DATE_COL].dt.quarter
    df["year"]    = df[_DATE_COL].dt.year

    # Seasonal flags
    df["is_peak_month"] = df["month"].isin([3, 4]).astype(int)
    df["is_off_month"]  = df["month"].isin([9, 10]).astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    for lag in [1, 7, 14, 30, 60, 90]:
        df[f"lag_{lag}"] = df.groupby(_SKU_COL)[_TARGET_COL].shift(lag)

    # Rolling statistics
    for window in [30, 60, 90]:
        df[f"rolling_mean_{window}"] = (
            df.groupby(_SKU_COL)[_TARGET_COL]
            .transform(lambda x: x.rolling(window).mean().shift(1))
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(_SKU_COL)[_TARGET_COL]
            .transform(lambda x: x.rolling(window).std().shift(1))
        )

    # Trend features
    df["month_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]

    df["rolling_diff_30d"] = (
        df.groupby(_SKU_COL)["rolling_mean_30"]
        .transform(lambda x: x.shift(1) - x.shift(31))
    ).fillna(0)

    df["rolling_growth_30d"] = (
        df.groupby(_SKU_COL)["rolling_mean_30"]
        .transform(lambda x: x.shift(1) / (x.shift(31) + 1e-6))
    ).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 5)

    # EWM features
    for alpha in [0.3, 0.5, 0.7]:
        alpha_str = str(alpha).replace(".", "")
        df[f"ewm_a{alpha_str}"] = (
            df.groupby(_SKU_COL)[_TARGET_COL]
            .transform(lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
        )

    df = df.dropna().reset_index(drop=True)
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [_DATE_COL, _TARGET_COL]]