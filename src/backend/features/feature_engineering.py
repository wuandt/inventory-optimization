"""Build the demand features used by the LightGBM forecast.

The key rule is simple: a feature for today may use today's calendar, but it
may only use demand from previous days.  This prevents target leakage.
"""

import json
import os

import numpy as np
import pandas as pd

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _config = json.load(f)

_TARGET_COL = _config["data"]["target_col"]
_DATE_COL = _config["data"]["date_col"]
_SKU_COL = _config["data"]["sku_col"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of calendar and demand-history features per SKU-date.

    Early rows do not have enough history for every lag or rolling feature.
    Those missing values are kept because LightGBM can handle them.
    """
    data = df.copy()
    data = data.sort_values([_SKU_COL, _DATE_COL]).reset_index(drop=True)

    # Step 1: add calendar information that is already known in advance.
    data["month"] = data[_DATE_COL].dt.month
    data["quarter"] = data[_DATE_COL].dt.quarter
    data["year"] = data[_DATE_COL].dt.year
    data["day_of_week"] = data[_DATE_COL].dt.dayofweek
    data["day_of_month"] = data[_DATE_COL].dt.day
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_month_start"] = (data["day_of_month"] <= 5).astype(int)
    data["is_month_end"] = (data["day_of_month"] >= 25).astype(int)

    # Sine and cosine tell the model that December is close to January and
    # Sunday is close to Monday.
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    demand_by_sku = data.groupby(_SKU_COL, observed=True)[_TARGET_COL]

    # Step 2: add demand from specific earlier days.
    for lag in [1, 7, 14, 30, 60, 90]:
        data[f"lag_{lag}"] = demand_by_sku.shift(lag)

    # Step 3: summarize demand over the previous 30, 60 and 90 days.
    # shift(1) is important: today's demand is never included.
    for window in [30, 60, 90]:
        rolling_mean = demand_by_sku.transform(
            lambda sku_demand: sku_demand.shift(1)
            .rolling(window, min_periods=window)
            .mean()
        )
        rolling_standard_deviation = demand_by_sku.transform(
            lambda sku_demand: sku_demand.shift(1)
            .rolling(window, min_periods=window)
            .std()
        )
        data[f"rolling_mean_{window}"] = rolling_mean
        data[f"rolling_std_{window}"] = rolling_standard_deviation
        data[f"rolling_cv_{window}"] = rolling_standard_deviation / (
            rolling_mean + 1e-6
        )

    # Step 4: compare short-term and long-term averages to show demand trend.
    data["rolling_diff_30_60"] = (
        data["rolling_mean_30"] - data["rolling_mean_60"]
    )
    data["rolling_diff_30_90"] = (
        data["rolling_mean_30"] - data["rolling_mean_90"]
    )
    data["rolling_diff_60_90"] = (
        data["rolling_mean_60"] - data["rolling_mean_90"]
    )
    data["rolling_growth_30_60"] = data["rolling_mean_30"] / (
        data["rolling_mean_60"] + 1e-6
    )
    data["rolling_growth_30_90"] = data["rolling_mean_30"] / (
        data["rolling_mean_90"] + 1e-6
    )
    data["rolling_growth_60_90"] = data["rolling_mean_60"] / (
        data["rolling_mean_90"] + 1e-6
    )

    # Step 5: add weighted averages. A larger alpha gives more weight to
    # recent demand.
    for alpha in [0.3, 0.5, 0.7]:
        alpha_name = str(alpha).replace(".", "")
        data[f"ewm_a{alpha_name}"] = demand_by_sku.transform(
            lambda sku_demand: sku_demand.shift(1)
            .ewm(alpha=alpha, adjust=False)
            .mean()
        )

    data["month_index"] = (
        (data["year"] - data["year"].min()) * 12 + data["month"]
    )
    return data


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model-input columns in their existing dataframe order."""
    feature_columns = []
    excluded_columns = {_DATE_COL, _TARGET_COL}

    for column in df.columns:
        if column not in excluded_columns:
            feature_columns.append(column)

    return feature_columns