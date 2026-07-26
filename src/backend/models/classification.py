"""Classify SKUs and choose which classes need an inventory-policy change.

Only information available before policy calibration is used.  This keeps the
ABC-XYZ labels and optimization scope independent from the final test results.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _config = json.load(f)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_DATE_COL = _config["data"]["date_col"]
_SKU_COL = _config["data"]["sku_col"]
_TARGET_COL = _config["data"]["target_col"]
_VALIDATION_START = pd.Timestamp(_config["data"]["validation_start"])
_MODEL_SELECTION_START = pd.Timestamp(
    _config["data"].get("model_selection_start", _VALIDATION_START)
)
_MODEL_SELECTION_END = pd.Timestamp(
    _config["data"].get("model_selection_end", _config["data"]["validation_end"])
)
_POLICY_CALIBRATION_START = pd.Timestamp(
    _config["data"].get(
        "policy_calibration_start", _config["data"]["final_test_start"]
    )
)
_ABC_A = _config["abc_xyz"]["abc_a_threshold"]
_ABC_B = _config["abc_xyz"]["abc_b_threshold"]
_XYZ_Q1 = _config["abc_xyz"]["xyz_q1"]
_XYZ_Q3 = _config["abc_xyz"]["xyz_q3"]
_XYZ_SEASON_LENGTH = _config["abc_xyz"]["xyz_season_length"]
_ABC_VALUE_BASIS = _config["abc_xyz"].get("abc_value_basis", "revenue")
_FOLD_DAYS = _config["backtest"]["validation_fold_days"]
_N_FOLDS = _config["backtest"]["validation_n_folds"]
_SCOPE = _config["optimization_scope"]


def _before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return rows observed before a cutoff date."""
    rows_before_cutoff = df[_DATE_COL] < cutoff
    return df.loc[rows_before_cutoff].copy()


def _demand_value(df: pd.DataFrame) -> pd.Series:
    """Convert unit demand to the economic value selected in the config.

    The project currently uses gross margin. Revenue and consumption cost are
    kept as optional alternatives for sensitivity analysis.
    """

    if _ABC_VALUE_BASIS == "gross_margin":
        unit_value = (df["unit_price"] - df["unit_cost"]).clip(lower=0)
    elif _ABC_VALUE_BASIS == "annual_consumption_cost":
        unit_value = df["unit_cost"]
    elif _ABC_VALUE_BASIS == "revenue":
        unit_value = df["unit_price"]
    else:
        raise ValueError(
            "abc_value_basis must be one of: gross_margin, "
            "annual_consumption_cost, revenue."
        )
    return df[_TARGET_COL] * unit_value


def classify_abc(
    df: pd.DataFrame, observed_until: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Assign A, B or C based on cumulative demand value.

    Demand is used instead of sales. A stockout can reduce sales, so using sales
    could incorrectly make an important SKU look less valuable.
    """
    if observed_until is None:
        cutoff_date = _POLICY_CALIBRATION_START
    else:
        cutoff_date = pd.Timestamp(observed_until)

    observed_rows = _before(df, cutoff_date)
    observed_rows["demand_value"] = _demand_value(observed_rows)

    abc_df = (
        observed_rows.groupby(_SKU_COL, observed=True)
        .agg(total_demand_value=("demand_value", "sum"))
        .reset_index()
        .sort_values("total_demand_value", ascending=False)
        .reset_index(drop=True)
    )
    total_value = abc_df["total_demand_value"].sum()
    if total_value <= 0:
        raise ValueError("ABC classification requires positive total demand value.")

    abc_df["cumulative_demand_value"] = abc_df["total_demand_value"].cumsum()
    abc_df["cumulative_demand_value_pct"] = (
        abc_df["cumulative_demand_value"] / total_value
    )
    is_class_a = abc_df["cumulative_demand_value_pct"] <= _ABC_A
    is_class_b = abc_df["cumulative_demand_value_pct"] <= _ABC_B
    abc_df["abc_class"] = np.select(
        [is_class_a, is_class_b],
        ["A", "B"],
        default="C",
    )
    return abc_df


def classify_xyz(
    df: pd.DataFrame,
    model_selection_start: pd.Timestamp | None = None,
    model_selection_end: pd.Timestamp | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, str]:
    """Assign X, Y or Z based on relative forecasting error.

    X means lower error within this portfolio and Z means higher error.  Every
    candidate forecast for a date uses demand only from earlier dates.
    """
    if model_selection_start is None:
        selection_start = _MODEL_SELECTION_START
    else:
        selection_start = pd.Timestamp(model_selection_start)

    if model_selection_end is None:
        selection_end = _MODEL_SELECTION_END
    else:
        selection_end = pd.Timestamp(model_selection_end)

    if selection_end >= _POLICY_CALIBRATION_START:
        raise ValueError(
            "XYZ model-selection dates must end before policy calibration."
        )

    available_rows = df[_DATE_COL] <= selection_end
    observed_rows = df.loc[available_rows].copy()
    if observed_rows.empty:
        raise ValueError("XYZ classification requires model-selection history.")

    candidates = observed_rows[[_SKU_COL, _DATE_COL, _TARGET_COL]].copy()
    candidates = candidates.sort_values([_SKU_COL, _DATE_COL]).reset_index(drop=True)
    demand_by_sku = candidates.groupby(_SKU_COL, observed=True)[_TARGET_COL]

    # All three methods use only demand known before the forecast date.
    candidates["Naive"] = demand_by_sku.shift(1)
    candidates["SeasonalNaive"] = demand_by_sku.shift(_XYZ_SEASON_LENGTH)
    candidates["HistoricMean"] = demand_by_sku.transform(
        lambda sku_demand: sku_demand.shift(1).expanding(min_periods=1).mean()
    )

    date_is_in_selection_window = (
        (candidates[_DATE_COL] >= selection_start)
        & (candidates[_DATE_COL] <= selection_end)
    )
    dates_in_window = candidates.loc[
        date_is_in_selection_window,
        _DATE_COL,
    ].unique()
    selection_dates = pd.DatetimeIndex(
        sorted(dates_in_window)
    )
    required_selection_days = _FOLD_DAYS * _N_FOLDS
    if len(selection_dates) < required_selection_days:
        raise ValueError(
            f"XYZ requires {required_selection_days} model-selection dates; "
            f"found {len(selection_dates)}."
        )

    dates_used_for_selection = selection_dates[-required_selection_days:]
    selection_rows = candidates[_DATE_COL].isin(dates_used_for_selection)
    selection = candidates.loc[selection_rows].copy()

    metric_rows = []
    for model_name in ["Naive", "SeasonalNaive", "HistoricMean"]:
        rows_with_forecast = selection.dropna(subset=[model_name])
        forecast_error = (
            rows_with_forecast[model_name] - rows_with_forecast[_TARGET_COL]
        )
        total_absolute_demand = rows_with_forecast[_TARGET_COL].abs().sum()

        metric_rows.append(
            {
                "model": model_name,
                "MAE": forecast_error.abs().mean(),
                "RMSE": np.sqrt((forecast_error**2).mean()),
                "WAPE": forecast_error.abs().sum() / total_absolute_demand,
                "Bias": forecast_error.mean(),
                "n_obs": len(rows_with_forecast),
            }
        )

    comparison = pd.DataFrame(metric_rows)
    comparison = comparison.sort_values("WAPE").reset_index(drop=True)
    selected_model = str(comparison.loc[0, "model"])

    # Calculate one normalized MAE for each SKU using the winning method.
    sku_error_rows = []
    for sku_id, sku_rows in selection.groupby(_SKU_COL, observed=True):
        mean_demand = sku_rows[_TARGET_COL].mean()
        if mean_demand <= 0:
            normalized_mae = np.nan
        else:
            mae = mean_absolute_error(
                sku_rows[_TARGET_COL],
                sku_rows[selected_model],
            )
            normalized_mae = mae / mean_demand

        sku_error_rows.append(
            {
                _SKU_COL: sku_id,
                "selection_nmae_selected_model": normalized_mae,
            }
        )

    xyz_df = pd.DataFrame(sku_error_rows)
    xyz_df = xyz_df.dropna(subset=["selection_nmae_selected_model"])
    if xyz_df.empty:
        raise ValueError(
            "XYZ classification could not calculate model-selection errors."
        )

    q1 = xyz_df["selection_nmae_selected_model"].quantile(_XYZ_Q1)
    q3 = xyz_df["selection_nmae_selected_model"].quantile(_XYZ_Q3)
    error_is_low = xyz_df["selection_nmae_selected_model"] < q1
    error_is_medium = xyz_df["selection_nmae_selected_model"] <= q3
    xyz_df["xyz_class"] = np.select(
        [error_is_low, error_is_medium],
        ["X", "Y"],
        default="Z",
    )
    xyz_df["selected_model"] = selected_model
    if return_diagnostics:
        return xyz_df, comparison, selected_model
    return xyz_df


def merge_abc_xyz(abc_df: pd.DataFrame, xyz_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the two labels, for example A + X becomes AX."""
    merged = abc_df[[_SKU_COL, "abc_class"]].merge(
        xyz_df[[_SKU_COL, "xyz_class"]],
        on=_SKU_COL,
        how="inner",
        validate="one_to_one",
    )
    merged["class"] = merged["abc_class"] + merged["xyz_class"]
    return merged[[_SKU_COL, "class"]]


def compute_sku_metric(
    df: pd.DataFrame, observed_until: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Calculate inventory, demand, service and variability by SKU."""
    if observed_until is not None:
        metric_data = _before(df, pd.Timestamp(observed_until))
    else:
        metric_data = df.copy()

    sku_metric = (
        metric_data.groupby(_SKU_COL, observed=True)
        .agg(
            avg_inventory=("inventory_level", "mean"),
            total_demand=(_TARGET_COL, "sum"),
            total_sales=("sales_quantity", "sum"),
            num_days=(_DATE_COL, "nunique"),
        )
        .reset_index()
    )
    sku_metric["daily_demand"] = sku_metric["total_demand"] / sku_metric["num_days"]
    sku_metric["DOI"] = (
        sku_metric["avg_inventory"] / sku_metric["daily_demand"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    sku_metric["fill_rate"] = (
        sku_metric["total_sales"] / sku_metric["total_demand"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    sku_metric["lost_sales"] = (
        sku_metric["total_demand"] - sku_metric["total_sales"]
    ).clip(
        lower=0
    )

    demand_variability = (
        metric_data.groupby(_SKU_COL, observed=True)[_TARGET_COL]
        .agg(["mean", "std"])
        .reset_index()
    )
    demand_variability["CV"] = (
        demand_variability["std"] / demand_variability["mean"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return sku_metric.merge(
        demand_variability[[_SKU_COL, "CV"]],
        on=_SKU_COL,
        validate="one_to_one",
    )


def select_optimization_scope(
    df: pd.DataFrame, sku_class_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose classes that need protection, understock or overstock action."""
    observed_rows = _before(df, _POLICY_CALIBRATION_START)
    observed_rows = observed_rows.merge(
        sku_class_df,
        on=_SKU_COL,
        validate="many_to_one",
    )
    observed_rows["demand_value"] = _demand_value(observed_rows)
    observed_rows["lost"] = (
        observed_rows[_TARGET_COL] - observed_rows["sales_quantity"]
    ).clip(lower=0)

    summary = (
        observed_rows.groupby("class", observed=True)
        .agg(
            sku_count=(_SKU_COL, "nunique"),
            demand_value=("demand_value", "sum"),
            total_demand=(_TARGET_COL, "sum"),
            total_sales=("sales_quantity", "sum"),
            lost_sales=("lost", "sum"),
            days=(_DATE_COL, "nunique"),
        )
        .reset_index()
    )

    # First total inventory by class and date, then average the daily totals.
    daily_inventory_by_class = observed_rows.groupby(
        ["class", _DATE_COL],
        observed=True,
    )["inventory_level"].sum()
    average_inventory_by_class = (
        daily_inventory_by_class.groupby("class").mean()
    )
    summary["avg_inventory"] = summary["class"].map(
        average_inventory_by_class
    )
    summary["fill_rate"] = summary["total_sales"] / summary["total_demand"]
    summary["lost_sales_per_day"] = summary["lost_sales"] / summary["days"]
    daily_demand = summary["total_demand"] / summary["days"]
    summary["DOI"] = summary["avg_inventory"] / daily_demand
    summary["abc"] = summary["class"].str[0]

    a_value_total = summary.loc[summary["abc"] == "A", "demand_value"].sum()
    if a_value_total <= 0:
        raise ValueError("Optimization scope requires positive A-class demand value.")

    summary["a_demand_value_share"] = np.where(
        summary["abc"] == "A",
        summary["demand_value"] / a_value_total,
        np.nan,
    )
    doi_threshold = summary["DOI"].quantile(_SCOPE["overstock_doi_quantile"])

    strategic_rows = (
        (summary["abc"] == "A")
        & (summary["a_demand_value_share"] >= _SCOPE["high_value_a_share_min"])
    )
    strategic = summary.loc[strategic_rows].copy()
    strategic["intervention"] = "protect_strategic_value"

    understock_rows = (
        (summary["abc"] == "C")
        & (summary["fill_rate"] < _SCOPE["understock_fill_rate_max"])
    )
    understock = summary.loc[understock_rows].copy()
    understock["intervention"] = "correct_understock"

    overstock_rows = (
        (summary["fill_rate"] >= _SCOPE["overstock_fill_rate_min"])
        & (summary["DOI"] >= doi_threshold)
    )
    overstock = summary.loc[overstock_rows].copy()
    overstock["intervention"] = "reduce_overstock"

    scope_parts = [strategic, understock, overstock]
    scope = pd.concat(scope_parts, ignore_index=True)
    scope = scope.drop_duplicates("class")
    return scope, summary


def save_sku_class(sku_class_df: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    sku_class_df.to_csv(os.path.join(_PRO_DIR, "sku_class.csv"), index=False)


def save_sku_metric(sku_metric_df: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    sku_metric_df.to_csv(os.path.join(_PRO_DIR, "sku_metric.csv"), index=False)


def save_optimization_scope(scope_df: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    scope_df.to_csv(os.path.join(_PRO_DIR, "optimization_scope.csv"), index=False)