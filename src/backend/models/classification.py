import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta, Naive, SeasonalNaive

# ── Load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH) as f:
    _config = json.load(f)

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_DATE_COL = _config["data"]["date_col"]
_SKU_COL  = _config["data"]["sku_col"]

_SPLIT_DATE    = _config["data"]["split_date"]
_ABC_A         = _config["abc_xyz"]["abc_a_threshold"]
_ABC_B         = _config["abc_xyz"]["abc_b_threshold"]
_XYZ_Q1        = _config["abc_xyz"]["xyz_q1"]
_XYZ_Q3        = _config["abc_xyz"]["xyz_q3"]
_SEASON_LENGTH = _config["abc_xyz"]["xyz_season_length"]


# ── ABC classification ─────────────────────────────────────────────────────────
def classify_abc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"] = df["sales_quantity"] * df["unit_price"]

    abc_df = (
        df.groupby(_SKU_COL)
        .agg(total_revenue=("revenue", "sum"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    abc_df["cumulative_revenue"]     = abc_df["total_revenue"].cumsum()
    abc_df["cumulative_revenue_pct"] = abc_df["cumulative_revenue"] / abc_df["total_revenue"].sum()

    def _abc(x):
        if x <= _ABC_A:
            return "A"
        elif x <= _ABC_B:
            return "B"
        return "C"

    abc_df["abc_class"] = abc_df["cumulative_revenue_pct"].apply(_abc)
    return abc_df

def classify_xyz(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df[_DATE_COL] < _SPLIT_DATE].copy()
    test  = df[df[_DATE_COL] >= _SPLIT_DATE].copy()

    sf_train = train[[_SKU_COL, _DATE_COL, "demand"]].rename(
        columns={_SKU_COL: "unique_id", _DATE_COL: "ds", "demand": "y"}
    )
    sf_test = test[[_SKU_COL, _DATE_COL, "demand"]].rename(
        columns={_SKU_COL: "unique_id", _DATE_COL: "ds", "demand": "y"}
    )
    horizon = sf_test["ds"].nunique()

    sf = StatsForecast(
        models=[Naive(), SeasonalNaive(season_length=_SEASON_LENGTH), AutoTheta(season_length=_SEASON_LENGTH)],
        freq="D",
        n_jobs=-1,
    )
    sf.fit(sf_train)
    preds = sf.predict(h=horizon)

    result = preds.merge(sf_test, on=["unique_id", "ds"], how="left")

    def _mae_pct(group):
        mae = mean_absolute_error(group["y"], group["AutoTheta"])
        return mae / group["y"].mean()

    xyz_df = (
        result.groupby("unique_id")
        .apply(_mae_pct)
        .reset_index()
    )
    xyz_df.columns = [_SKU_COL, "%mae_autotheta"]

    q1 = xyz_df["%mae_autotheta"].quantile(_XYZ_Q1)
    q3 = xyz_df["%mae_autotheta"].quantile(_XYZ_Q3)

    def _xyz(val):
        if val < q1:
            return "X"
        elif val <= q3:
            return "Y"
        return "Z"

    xyz_df["xyz_class"] = xyz_df["%mae_autotheta"].apply(_xyz)
    return xyz_df

def merge_abc_xyz(abc_df: pd.DataFrame, xyz_df: pd.DataFrame) -> pd.DataFrame:
    df = abc_df[[_SKU_COL, "abc_class"]].merge(
        xyz_df[[_SKU_COL, "xyz_class"]], on=_SKU_COL, how="left"
    )
    df["class"] = df["abc_class"] + df["xyz_class"]
    return df[[_SKU_COL, "class"]]

def compute_sku_metric(df: pd.DataFrame) -> pd.DataFrame:
    sku_metric = (
        df.groupby(_SKU_COL)
        .agg(
            avg_inventory =("inventory_level", "mean"),
            total_demand  =("demand",          "sum"),
            total_sales   =("sales_quantity",  "sum"),
            num_days      =(_DATE_COL,         "nunique"),
        )
        .reset_index()
    )

    sku_metric["daily_demand"] = sku_metric["total_demand"] / sku_metric["num_days"]
    sku_metric["DOI"]          = (sku_metric["avg_inventory"] / sku_metric["daily_demand"]).replace([np.inf, -np.inf], np.nan)
    sku_metric["fill_rate"]    = (sku_metric["total_sales"]   / sku_metric["total_demand"]).replace([np.inf, -np.inf], np.nan)
    sku_metric["lost_sales"]   = (sku_metric["total_demand"]  - sku_metric["total_sales"]).replace([np.inf, -np.inf], np.nan)

    cv_df = (
        df.groupby(_SKU_COL)["demand"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_df["CV"] = (cv_df["std"] / cv_df["mean"]).replace([np.inf, -np.inf], np.nan)

    return sku_metric.merge(cv_df[[_SKU_COL, "CV"]], on=_SKU_COL)

def save_sku_class(sku_class_df: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    path = os.path.join(_PRO_DIR, "sku_class.csv")
    sku_class_df.to_csv(path, index=False)
    print(f"Saved → {path}")

def save_sku_metric(sku_metric_df: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    path = os.path.join(_PRO_DIR, "sku_metric.csv")
    sku_metric_df.to_csv(path, index=False)
    print(f"Saved → {path}")