import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH) as f:
    _config = json.load(f)

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_DATE_COL    = _config["data"]["date_col"]
_SKU_COL     = _config["data"]["sku_col"]
_SL_MAP      = _config["inventory"]["service_level_map"]
_ORDER_COST  = _config["inventory"]["ordering_cost"]
_HOLD_RATE   = _config["inventory"]["holding_rate"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_policy_params(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, sku_class_df: pd.DataFrame) -> pd.DataFrame:
    forecast_df = forecast_df.copy()
    forecast_df["squared_error"] = (forecast_df["forecast"] - forecast_df["demand"]) ** 2

    forecast_agg = (
        forecast_df.groupby(_SKU_COL)
        .agg(
            rmse         =("squared_error", lambda x: np.sqrt(x.mean())),
            avg_forecast =("forecast",      "mean"),
        )
        .reset_index()
    )

    inventory_agg = (
        inventory_df.groupby(_SKU_COL)
        .agg(
            lead_time     =("lead_time",     "mean"),
            unit_cost     =("unit_cost",     "mean"),
            unit_price    =("unit_price",    "mean"),
            order_quantity=("order_quantity","mean"),
        )
        .reset_index()
    )

    policy = (
        forecast_agg
        .merge(inventory_agg,  on=_SKU_COL)
        .merge(sku_class_df,   on=_SKU_COL)
    )

    policy["target_service_level"] = policy["class"].map(_SL_MAP)
    policy["z_score"]              = norm.ppf(policy["target_service_level"] / 100)
    policy["safety_stock"]         = policy["z_score"] * policy["rmse"] * (policy["lead_time"] ** 0.5)
    policy["rop"]                  = policy["avg_forecast"] * policy["lead_time"] + policy["safety_stock"]

    return policy[[
        _SKU_COL, "class", "lead_time", "unit_cost", "unit_price",
        "order_quantity", "target_service_level", "safety_stock", "rop",
    ]]

def compute_old_policy_metric(inventory_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    days = inventory_df[_DATE_COL].nunique()
    target_skus = policy_df[_SKU_COL].unique()

    old = (
        inventory_df[inventory_df[_SKU_COL].isin(target_skus)]
        .merge(policy_df[[_SKU_COL, "class"]], on=_SKU_COL)
        .copy()
    )

    old["lost_sales"]   = (old["demand"] - old["sales_quantity"]).clip(lower=0)
    old["order_placed"] = (old["order_received"] > 0).astype(int)

    old_policy = (
        old.groupby([_SKU_COL, "class"])
        .agg(
            total_demand  =("demand",          "sum"),
            total_sales   =("sales_quantity",  "sum"),
            avg_inventory =("inventory_level", "mean"),
            total_lost    =("lost_sales",      "sum"),
            num_orders    =("order_placed",    "sum"),
            unit_cost     =("unit_cost",       "mean"),
            unit_price    =("unit_price",      "mean"),
        )
        .reset_index()
    )

    old_policy["fill_rate"]    = old_policy["total_sales"]  / old_policy["total_demand"]
    old_policy["turnover"]     = old_policy["total_demand"] / old_policy["avg_inventory"]
    old_policy["DOI"]          = old_policy["avg_inventory"] / (old_policy["total_demand"] / days)
    old_policy["stockout_cost"]= old_policy["total_lost"]   * (old_policy["unit_price"] - old_policy["unit_cost"])
    old_policy["holding_cost"] = old_policy["avg_inventory"]* old_policy["unit_cost"] * _HOLD_RATE * (days / 365)
    old_policy["ordering_cost"]= old_policy["num_orders"]   * _ORDER_COST
    old_policy["total_cost"]   = old_policy["holding_cost"] + old_policy["ordering_cost"] + old_policy["stockout_cost"]

    return old_policy[[
        _SKU_COL, "class", "total_demand", "total_sales", "avg_inventory",
        "fill_rate", "turnover", "DOI", "stockout_cost", "holding_cost",
        "ordering_cost", "total_cost",
    ]]

def _simulate_sku(daily_data: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    inventory      = daily_data["inventory_level"].iloc[0]
    rop            = policy["rop"]
    order_qty      = policy["order_quantity"]
    lead_time      = int(policy["lead_time"])
    pending_orders = {}
    results        = []

    for i, row in daily_data.iterrows():
        if i in pending_orders:
            inventory += pending_orders.pop(i)

        demand    = row["demand"]
        sales     = min(demand, inventory)
        lost      = demand - sales
        inventory -= sales

        inventory_position = inventory + sum(pending_orders.values())
        order_placed = 0
        if inventory_position <= rop:
            arrival_idx = i + lead_time
            pending_orders[arrival_idx] = pending_orders.get(arrival_idx, 0) + order_qty
            order_placed = 1

        results.append({
            _SKU_COL            : policy[_SKU_COL],
            _DATE_COL           : row[_DATE_COL],
            "inventory"         : round(inventory, 4),
            "inventory_position": round(inventory_position, 4),
            "demand"            : demand,
            "sales_quantity"    : sales,
            "lost"              : lost,
            "order_placed"      : order_placed,
        })

    return pd.DataFrame(results)

def simulate_new_policy(inventory_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    target_skus = policy_df[_SKU_COL].unique()
    sim_data    = (
        inventory_df[inventory_df[_SKU_COL].isin(target_skus)]
        .sort_values([_SKU_COL, _DATE_COL])
        .reset_index(drop=True)
    )

    all_results = []
    for sku_id, group in sim_data.groupby(_SKU_COL):
        policy = policy_df[policy_df[_SKU_COL] == sku_id].iloc[0]
        result = _simulate_sku(group.reset_index(), policy)
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)

def compute_new_policy_metric(sim_results: pd.DataFrame, policy_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    days = inventory_df[_DATE_COL].nunique()

    new_policy = (
        sim_results.groupby(_SKU_COL)
        .agg(
            total_demand  =("demand",         "sum"),
            total_sales   =("sales_quantity", "sum"),
            total_lost    =("lost",           "sum"),
            avg_inventory =("inventory",      "mean"),
            num_orders    =("order_placed",   "sum"),
        )
        .reset_index()
    )

    new_policy = new_policy.merge(
        policy_df[[_SKU_COL, "class", "unit_cost", "unit_price"]], on=_SKU_COL
    )

    new_policy["fill_rate"]    = new_policy["total_sales"]  / new_policy["total_demand"]
    new_policy["turnover"]     = new_policy["total_demand"] / new_policy["avg_inventory"]
    new_policy["DOI"]          = new_policy["avg_inventory"] / (new_policy["total_demand"] / days)
    new_policy["stockout_cost"]= new_policy["total_lost"]   * (new_policy["unit_price"] - new_policy["unit_cost"])
    new_policy["holding_cost"] = new_policy["avg_inventory"]* new_policy["unit_cost"] * _HOLD_RATE * (days / 365)
    new_policy["ordering_cost"]= new_policy["num_orders"]   * _ORDER_COST
    new_policy["total_cost"]   = new_policy["holding_cost"] + new_policy["ordering_cost"] + new_policy["stockout_cost"]

    return new_policy[[
        _SKU_COL, "class", "total_demand", "total_sales", "avg_inventory",
        "fill_rate", "turnover", "DOI", "stockout_cost", "holding_cost",
        "ordering_cost", "total_cost",
    ]]

def save_simulation_results(policy_sku: pd.DataFrame, old_policy_metric: pd.DataFrame, new_policy_metric: pd.DataFrame) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)

    policy_sku.to_csv(os.path.join(_PRO_DIR, "policy_sku.csv"), index=False)
    old_policy_metric.to_csv(os.path.join(_PRO_DIR, "old_policy_metric.csv"), index=False)
    new_policy_metric.to_csv(os.path.join(_PRO_DIR, "new_policy_metric.csv"), index=False)

    logger.info("Simulation results saved → data/processed/")