"""Policy calibration and fair locked-evaluation inventory simulation.

Policies are calibrated from dedicated policy-calibration forecasts only. The
recorded historical policy and proposed policy are then run through the same
lost-sales simulator over the locked evaluation window using identical demand
paths and initial on-hand inventory. Receipt-inferred historical parameters are
kept only as an explicitly labeled sensitivity case.
"""

import json
import logging
import os
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _config = json.load(f)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_DATE_COL = _config["data"]["date_col"]
_SKU_COL = _config["data"]["sku_col"]
_VALIDATION_START = pd.Timestamp(_config["data"]["validation_start"])
_POLICY_CALIBRATION_START = pd.Timestamp(
    _config["data"].get("policy_calibration_start", _VALIDATION_START)
)
_POLICY_CALIBRATION_END = pd.Timestamp(
    _config["data"].get(
        "policy_calibration_end",
        _config["data"].get("validation_end", _config["data"]["final_test_start"]),
    )
)
_FINAL_TEST_START = pd.Timestamp(_config["data"]["final_test_start"])
_SL_MAP = _config["inventory"]["service_level_map"]
_SL_BY_INTERVENTION = _config["inventory"]["service_level_by_intervention"]
_ORDER_COST = _config["inventory"]["ordering_cost"]
_HOLD_RATE = _config["inventory"]["holding_rate"]
_SHORTAGE_COST_MULTIPLIER = _config["inventory"].get(
    "shortage_cost_multiplier", 1.0
)

logger = logging.getLogger(__name__)


def _positive_integer_lead_time(value: object, *, context: str) -> int:
    """Check lead time and return it as a whole number of days."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: lead_time must be a positive integer.") from exc

    is_invalid = (
        not np.isfinite(numeric)
        or numeric <= 0
        or not numeric.is_integer()
    )
    if is_invalid:
        raise ValueError(f"{context}: lead_time must be a positive integer.")
    return int(numeric)


def _validated_daily_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Sort and validate one SKU's complete daily calendar."""
    required = set(required_columns)
    required.add(_DATE_COL)
    available = set(frame.columns)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    result = frame.copy()
    result[_DATE_COL] = pd.to_datetime(result[_DATE_COL], errors="raise")
    if result[_DATE_COL].isna().any():
        raise ValueError(f"{label} contains missing dates.")
    if result.duplicated(_DATE_COL).any():
        raise ValueError(f"{label} contains duplicate dates.")
    result = result.sort_values(_DATE_COL).reset_index(drop=True)
    gaps = result[_DATE_COL].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            f"{label} must contain one consecutive row per calendar day."
        )
    return result


def _validated_calibration_forecasts(
    validation_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that calibration rows are causal forecasts outside final holdout."""
    if validation_forecast_df.empty:
        raise ValueError("Policy calibration requires validation forecasts.")
    required = {_SKU_COL, _DATE_COL, "demand", "forecast"}
    missing = sorted(required - set(validation_forecast_df.columns))
    if missing:
        raise ValueError(f"Policy calibration forecasts are missing columns: {missing}")

    calibration = validation_forecast_df.copy()
    calibration[_DATE_COL] = pd.to_datetime(
        calibration[_DATE_COL], errors="raise"
    )
    if (
        calibration[_DATE_COL].min() < _POLICY_CALIBRATION_START
        or calibration[_DATE_COL].max() > _POLICY_CALIBRATION_END
        or (calibration[_DATE_COL] >= _FINAL_TEST_START).any()
    ):
        raise ValueError(
            "Policy calibration forecasts must stay inside the configured "
            f"{_POLICY_CALIBRATION_START.date()} to "
            f"{_POLICY_CALIBRATION_END.date()} window."
        )
    if calibration.duplicated([_SKU_COL, _DATE_COL]).any():
        raise ValueError("Policy calibration forecasts contain duplicate SKU-date rows.")
    for column in ["demand", "forecast"]:
        values = pd.to_numeric(calibration[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Policy calibration {column} must be finite numeric values.")
        calibration[column] = values.astype(float)
    if (calibration["demand"] < 0).any():
        raise ValueError("Policy calibration demand must be nonnegative.")

    groups = []
    for sku_id, group in calibration.groupby(_SKU_COL, observed=True):
        groups.append(
            _validated_daily_frame(
                group,
                label=f"Calibration forecast for {sku_id}",
                required_columns=["demand", "forecast"],
            )
        )
    return pd.concat(groups, ignore_index=True)


def _pre_holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[_DATE_COL] < _FINAL_TEST_START].copy()


def _final_holdout(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[_DATE_COL] >= _FINAL_TEST_START].copy()


def compute_policy_params(
    validation_forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sku_class_df: pd.DataFrame,
    optimization_scope_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calibrate policy from validation residuals, never evaluation residuals."""
    validation_forecast_df = _validated_calibration_forecasts(
        validation_forecast_df
    )

    history = _pre_holdout(inventory_df)
    inventory_agg = (
        history.groupby(_SKU_COL, observed=True)
        .agg(
            lead_time=("lead_time", "median"),
            unit_cost=("unit_cost", "median"),
            unit_price=("unit_price", "median"),
            order_quantity=("order_quantity", "median"),
        )
        .reset_index()
    )
    calibration = validation_forecast_df.merge(
        inventory_agg[[_SKU_COL, "lead_time"]],
        on=_SKU_COL,
        validate="many_to_one",
    ).sort_values([_SKU_COL, _DATE_COL])
    calibration["residual"] = calibration["demand"] - calibration["forecast"]
    calibration["squared_error"] = calibration["residual"] ** 2
    lead_time_error_by_sku: dict[str, np.ndarray] = {}
    forecast_rows = []
    for sku_id, group in calibration.groupby(_SKU_COL, observed=True):
        lead_time = _positive_integer_lead_time(
            group["lead_time"].iloc[0],
            context=f"Policy calibration for {sku_id}",
        )
        cumulative_error = (
            group["residual"].rolling(lead_time, min_periods=lead_time).sum().dropna()
        )
        if cumulative_error.empty:
            raise ValueError(f"Insufficient validation residuals for lead time of {sku_id}.")
        lead_time_error_by_sku[sku_id] = cumulative_error.to_numpy(dtype=float)
        forecast_rows.append(
            {
                _SKU_COL: sku_id,
                "validation_rmse": np.sqrt(group["squared_error"].mean()),
                "avg_forecast": group["forecast"].mean(),
                "validation_bias": (group["forecast"] - group["demand"]).mean(),
                "lead_time_error_std": cumulative_error.std(ddof=1),
                "lead_time_error_samples": len(cumulative_error),
            }
        )
    forecast_agg = pd.DataFrame(forecast_rows)
    policy = (
        forecast_agg.merge(inventory_agg, on=_SKU_COL, validate="one_to_one")
        .merge(sku_class_df, on=_SKU_COL, validate="one_to_one")
    )
    if optimization_scope_df is not None:
        policy = policy.merge(
            optimization_scope_df[["class", "intervention"]].drop_duplicates("class"),
            on="class",
            how="left",
            validate="many_to_one",
        )
    else:
        policy["intervention"] = np.nan
    # A class-specific SLA is the most precise business rule.  The intervention
    # target is only a fallback for a dynamically selected class that has no
    # explicit SLA in configuration.
    policy["target_service_level"] = policy["class"].map(_SL_MAP).fillna(
        policy["intervention"].map(_SL_BY_INTERVENTION)
    )
    if policy["target_service_level"].isna().any():
        unresolved = policy.loc[policy["target_service_level"].isna(), "class"].unique().tolist()
        raise ValueError(f"No service-level rule for selected classes: {unresolved}")
    policy["z_score"] = norm.ppf(policy["target_service_level"] / 100)
    # Calculate each SKU separately so it is easy to see which residual history
    # and service target are used for its safety stock.
    safety_stock_values = []
    for _, row in policy.iterrows():
        sku_id = row[_SKU_COL]
        service_probability = row["target_service_level"] / 100
        residual_history = lead_time_error_by_sku[sku_id]
        residual_quantile = float(
            np.quantile(residual_history, service_probability)
        )
        safety_stock_values.append(max(0.0, residual_quantile))

    policy["safety_stock"] = safety_stock_values
    policy["rop"] = policy["avg_forecast"] * policy["lead_time"] + policy["safety_stock"]
    return policy[
        [
            _SKU_COL,
            "class",
            "lead_time",
            "unit_cost",
            "unit_price",
            "order_quantity",
            "target_service_level",
            "validation_rmse",
            "validation_bias",
            "lead_time_error_std",
            "lead_time_error_samples",
            "safety_stock",
            "rop",
        ]
    ]


def estimate_historical_policy(
    inventory_df: pd.DataFrame, sku_class_df: pd.DataFrame, target_skus: pd.Series
) -> pd.DataFrame:
    """Return the recorded pre-holdout historical (s, Q) policy.

    ``reorder_point`` is part of the raw data contract, so it is the primary
    historical-policy source.  Receipt-inferred ROP remains available through
    :func:`estimate_inferred_historical_policy_sensitivity` for an explicit
    baseline-sensitivity analysis rather than silently replacing recorded data.
    """
    history = _pre_holdout(inventory_df)
    history = history[history[_SKU_COL].isin(target_skus)].copy()
    required = {
        _SKU_COL,
        "reorder_point",
        "order_quantity",
        "lead_time",
        "unit_cost",
        "unit_price",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Recorded historical policy is missing columns: {missing}")
    if history.empty:
        raise ValueError("Recorded historical policy has no pre-holdout rows.")

    rows = []
    for sku_id, group in history.groupby(_SKU_COL, observed=True):
        lead_time = _positive_integer_lead_time(
            group["lead_time"].median(),
            context=f"Recorded historical policy for {sku_id}",
        )
        rop = group["reorder_point"].median()
        quantity = group["order_quantity"].median()
        if not np.isfinite(rop) or rop < 0:
            raise ValueError(
                f"Recorded historical policy for {sku_id}: reorder_point "
                "must be finite and nonnegative."
            )
        if not np.isfinite(quantity) or quantity <= 0:
            raise ValueError(
                f"Recorded historical policy for {sku_id}: order_quantity "
                "must be finite and positive."
            )
        rows.append(
            {
                _SKU_COL: sku_id,
                "rop": float(rop),
                "order_quantity": float(quantity),
                "lead_time": lead_time,
                "unit_cost": group["unit_cost"].median(),
                "unit_price": group["unit_price"].median(),
            }
        )

    requested = set(pd.Series(target_skus).dropna())
    found = set()
    for row in rows:
        found.add(row[_SKU_COL])
    missing_skus = sorted(requested - found)
    if missing_skus:
        raise ValueError(
            f"Recorded historical policy is missing target SKUs: {missing_skus}"
        )
    estimated = pd.DataFrame(rows).merge(
        sku_class_df, on=_SKU_COL, validate="one_to_one"
    )
    return estimated[
        [_SKU_COL, "class", "lead_time", "unit_cost", "unit_price", "order_quantity", "rop"]
    ]


def estimate_inferred_historical_policy_sensitivity(
    inventory_df: pd.DataFrame,
    sku_class_df: pd.DataFrame,
    target_skus: pd.Series,
) -> pd.DataFrame:
    """Infer ROP from receipts for baseline sensitivity only.

    An order date is reconstructed as receipt date minus recorded lead time and
    its end-of-day inventory is used as an observable ROP proxy.  This cannot
    recover inventory position or outstanding orders, so callers should label
    it as a sensitivity case rather than the primary historical policy.
    """
    history = _pre_holdout(inventory_df)
    history = history[history[_SKU_COL].isin(target_skus)].copy()
    rows = []
    for sku_id, group in history.groupby(_SKU_COL, observed=True):
        group = group.sort_values(_DATE_COL)
        lead_time = _positive_integer_lead_time(
            group["lead_time"].median(),
            context=f"Receipt-inferred historical policy for {sku_id}",
        )
        receipts = group[group["order_received"] > 0].copy()
        quantity = receipts["order_received"].median()
        if pd.isna(quantity):
            quantity = group["order_quantity"].median()
        if not np.isfinite(quantity) or quantity <= 0:
            raise ValueError(
                f"Receipt-inferred historical policy for {sku_id}: "
                "order_quantity must be finite and positive."
            )
        receipt_dates = set(receipts[_DATE_COL])
        inferred_order_dates = set()
        for receipt_date in receipt_dates:
            estimated_order_date = receipt_date - pd.Timedelta(days=lead_time)
            inferred_order_dates.add(estimated_order_date)
        order_inventory = group[group[_DATE_COL].isin(inferred_order_dates)]["inventory_level"]
        rop = order_inventory.median()
        if pd.isna(rop):
            rop = group["inventory_level"].quantile(0.2)
        rows.append(
            {
                _SKU_COL: sku_id,
                "estimated_historical_rop": float(rop),
                "order_quantity": float(quantity),
                "lead_time": lead_time,
                "unit_cost": group["unit_cost"].median(),
                "unit_price": group["unit_price"].median(),
            }
        )
    estimated = pd.DataFrame(rows).merge(sku_class_df, on=_SKU_COL, validate="one_to_one")
    return estimated.rename(columns={"estimated_historical_rop": "rop"})[
        [_SKU_COL, "class", "lead_time", "unit_cost", "unit_price", "order_quantity", "rop"]
    ]


def _initial_inventory(inventory_df: pd.DataFrame, target_skus: pd.Series) -> pd.DataFrame:
    """Return end-of-day inventory immediately before locked evaluation."""
    history = _pre_holdout(inventory_df)
    target_history = history[history[_SKU_COL].isin(target_skus)]
    target_history = target_history.sort_values([_SKU_COL, _DATE_COL])
    last_rows = target_history.groupby(_SKU_COL, observed=True).tail(1)
    initial = last_rows[[_SKU_COL, "inventory_level"]].copy()
    return initial.rename(
        columns={"inventory_level": "initial_inventory"}
    )


def _simulate_sku(
    daily_demand: pd.DataFrame,
    policy: pd.Series,
    initial_inventory: float,
    forecast_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simulate daily inventory, sales, lost demand and purchase orders.

    A policy without ``forecast_schedule`` uses one fixed ROP. With a forecast
    schedule, the end-of-day decision uses tomorrow's forecast:

    ``tomorrow's forecast * lead time + safety stock``.

    The simulator does not place an order on the last day because that order
    cannot arrive inside the evaluation period.
    """
    sku_id = policy[_SKU_COL]
    daily_demand = _validated_daily_frame(
        daily_demand,
        label=f"Simulation demand for {sku_id}",
        required_columns=["demand"],
    )
    demand_values = pd.to_numeric(daily_demand["demand"], errors="coerce")
    if (
        demand_values.isna().any()
        or not np.isfinite(demand_values).all()
        or (demand_values < 0).any()
    ):
        raise ValueError(
            f"Simulation demand for {sku_id} must be finite and nonnegative."
        )
    daily_demand["demand"] = demand_values.astype(float)

    inventory = float(initial_inventory)
    if not np.isfinite(inventory) or inventory < 0:
        raise ValueError(
            f"Simulation for {sku_id}: initial_inventory must be finite and nonnegative."
        )
    static_rop = float(policy["rop"])
    if not np.isfinite(static_rop) or static_rop < 0:
        raise ValueError(
            f"Simulation for {sku_id}: rop must be finite and nonnegative."
        )
    order_qty = float(policy["order_quantity"])
    if not np.isfinite(order_qty) or order_qty <= 0:
        raise ValueError(
            f"Simulation for {sku_id}: order_quantity must be finite and positive."
        )
    lead_time = _positive_integer_lead_time(
        policy["lead_time"], context=f"Simulation for {sku_id}"
    )

    dynamic_rop_by_target_date: dict[pd.Timestamp, float] | None = None
    if forecast_schedule is not None:
        schedule = forecast_schedule.copy()
        if _SKU_COL in schedule.columns:
            schedule = schedule[schedule[_SKU_COL] == sku_id]
        schedule = _validated_daily_frame(
            schedule,
            label=f"Forecast schedule for {sku_id}",
            required_columns=["forecast"],
        )
        forecasts = pd.to_numeric(schedule["forecast"], errors="coerce")
        if (
            forecasts.isna().any()
            or not np.isfinite(forecasts).all()
            or (forecasts < 0).any()
        ):
            raise ValueError(
                f"Forecast schedule for {sku_id} must be finite and nonnegative."
            )
        safety_stock = float(policy.get("safety_stock", 0.0))
        if not np.isfinite(safety_stock) or safety_stock < 0:
            raise ValueError(
                f"Simulation for {sku_id}: safety_stock must be finite and nonnegative."
            )
        dynamic_rop_values = (
            forecasts.astype(float) * lead_time + safety_stock
        )
        dynamic_rop_by_target_date = {}
        for target_date, dynamic_rop in zip(
            schedule[_DATE_COL], dynamic_rop_values
        ):
            dynamic_rop_by_target_date[target_date] = dynamic_rop

    pending_orders: dict[int, float] = {}
    results = []
    last_period = len(daily_demand) - 1

    for period, (_, row) in enumerate(daily_demand.iterrows()):
        # Receive previously placed orders before serving today's demand.
        received = pending_orders.pop(period, 0.0)
        inventory += received

        demand = float(row["demand"])
        sales = min(demand, inventory)
        lost = demand - sales
        inventory -= sales

        terminal_period = period == last_period
        forecast_target_date = row[_DATE_COL] + pd.Timedelta(days=1)
        if terminal_period:
            rop_used = np.nan
            recorded_target_date = pd.NaT
        elif dynamic_rop_by_target_date is None:
            rop_used = static_rop
            recorded_target_date = pd.NaT
        else:
            if forecast_target_date not in dynamic_rop_by_target_date:
                raise ValueError(
                    f"Forecast schedule for {sku_id} is missing target date "
                    f"{forecast_target_date.date()} required by the end-of-day "
                    f"{row[_DATE_COL].date()} order decision."
                )
            rop_used = dynamic_rop_by_target_date[forecast_target_date]
            recorded_target_date = forecast_target_date

        # Inventory position includes stock that is already on order.
        inventory_position_before_order = inventory + sum(pending_orders.values())
        order_placed = 0
        can_place_order = not terminal_period
        if can_place_order and inventory_position_before_order <= rop_used:
            order_placed = 1
            arrival_period = period + lead_time
            existing_quantity = pending_orders.get(arrival_period, 0.0)
            pending_orders[arrival_period] = existing_quantity + order_qty

        inventory_position_after_order = inventory + sum(pending_orders.values())
        results.append(
            {
                _SKU_COL: sku_id,
                _DATE_COL: row[_DATE_COL],
                "initial_inventory": initial_inventory if period == 0 else np.nan,
                "received": received,
                "inventory": inventory,
                "inventory_position": inventory_position_after_order,
                "rop_used": rop_used,
                "forecast_target_date": recorded_target_date,
                "order_decision_eligible": can_place_order,
                "demand": demand,
                "sales_quantity": sales,
                "lost": lost,
                "order_placed": order_placed,
            }
        )
    return pd.DataFrame(results)


def simulate_policy(
    inventory_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    forecast_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every policy over locked evaluation with an optional dynamic ROP.

    Existing callers that omit ``forecast_schedule`` retain static ROP behavior.
    Dynamic schedules must contain ``sku_id``, ``date`` (the target date), and
    ``forecast``.  A final-day order is suppressed for both policies because it
    cannot affect any in-horizon service or inventory outcome.
    """
    target_skus = policy_df[_SKU_COL]
    if policy_df.duplicated(_SKU_COL).any():
        raise ValueError("Policy must contain exactly one row per SKU.")
    if forecast_schedule is not None:
        required = {_SKU_COL, _DATE_COL, "forecast"}
        missing = sorted(required - set(forecast_schedule.columns))
        if missing:
            raise ValueError(f"Dynamic forecast schedule is missing columns: {missing}")
    initial = _initial_inventory(inventory_df, target_skus)
    final_demand = _final_holdout(inventory_df)
    final_demand = final_demand[final_demand[_SKU_COL].isin(target_skus)]
    all_results = []
    for sku_id, group in final_demand.groupby(_SKU_COL, observed=True):
        policy = policy_df.loc[policy_df[_SKU_COL] == sku_id].iloc[0]
        initial_value = initial.loc[initial[_SKU_COL] == sku_id, "initial_inventory"]
        if initial_value.empty:
            raise ValueError(f"Missing pre-holdout initial inventory for {sku_id}.")
        sku_schedule = None
        if forecast_schedule is not None:
            sku_schedule = forecast_schedule[
                forecast_schedule[_SKU_COL] == sku_id
            ]
        # A combined full-portfolio policy can keep out-of-scope SKUs on their
        # recorded static ROP while applying dynamic forecasts only to the
        # intervention scope.
        if sku_schedule is not None and sku_schedule.empty:
            sku_schedule = None
        all_results.append(
            _simulate_sku(
                group.sort_values(_DATE_COL),
                policy,
                float(initial_value.iloc[0]),
                forecast_schedule=sku_schedule,
            )
        )
    simulated_skus = set()
    for result in all_results:
        if not result.empty:
            simulated_skus.add(result[_SKU_COL].iloc[0])
    missing_skus = sorted(set(target_skus) - simulated_skus)
    if missing_skus:
        raise ValueError(f"Locked evaluation demand is missing target SKUs: {missing_skus}")
    return pd.concat(all_results, ignore_index=True)


def compute_policy_metric(
    sim_results: pd.DataFrame,
    policy_df: pd.DataFrame,
    *,
    ordering_cost: float | None = None,
    holding_rate: float | None = None,
    stockout_cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute policy metrics under explicit, overridable cost assumptions."""
    ordering_cost = _ORDER_COST if ordering_cost is None else float(ordering_cost)
    holding_rate = _HOLD_RATE if holding_rate is None else float(holding_rate)
    stockout_cost_multiplier = float(stockout_cost_multiplier)
    assumptions = {
        "ordering_cost": ordering_cost,
        "holding_rate": holding_rate,
        "stockout_cost_multiplier": stockout_cost_multiplier,
    }
    invalid = []
    for name, value in assumptions.items():
        if not np.isfinite(value) or value < 0:
            invalid.append(name)
    if invalid:
        raise ValueError(
            f"Cost assumptions must be finite and nonnegative: {invalid}"
        )
    if sim_results.empty:
        raise ValueError("Policy metrics require non-empty simulation results.")

    days = sim_results[_DATE_COL].nunique()
    metric = (
        sim_results.groupby(_SKU_COL, observed=True)
        .agg(
            total_demand=("demand", "sum"),
            total_sales=("sales_quantity", "sum"),
            total_lost=("lost", "sum"),
            avg_inventory=("inventory", "mean"),
            num_orders=("order_placed", "sum"),
        )
        .reset_index()
        .merge(
            policy_df[[_SKU_COL, "class", "unit_cost", "unit_price"]],
            on=_SKU_COL,
            validate="one_to_one",
        )
    )
    metric["fill_rate"] = metric["total_sales"] / metric["total_demand"].replace(
        0, np.nan
    )
    # A receipt starts a new replenishment cycle. A cycle succeeds when no
    # demand is lost before the next receipt.
    simulation_with_cycles = sim_results.copy()
    receipt_flags = simulation_with_cycles["received"].gt(0)
    simulation_with_cycles["receipt_cycle"] = receipt_flags.groupby(
        simulation_with_cycles[_SKU_COL], observed=True
    ).cumsum()
    lost_demand_by_cycle = simulation_with_cycles.groupby(
        [_SKU_COL, "receipt_cycle"], observed=True
    )["lost"].sum()
    cycle_outcomes = lost_demand_by_cycle.eq(0)

    cycle_stats = cycle_outcomes.groupby(level=0).agg(["sum", "count"])
    cycle_stats = cycle_stats.rename(
        columns={
            "sum": "cycles_without_stockout",
            "count": "num_cycles",
        }
    )
    cycle_stats = cycle_stats.reset_index()
    metric = metric.merge(cycle_stats, on=_SKU_COL, validate="one_to_one")
    metric["cycle_service_level"] = (
        metric["cycles_without_stockout"] / metric["num_cycles"]
    )
    average_inventory = metric["avg_inventory"].replace(0, np.nan)
    average_daily_demand = (metric["total_demand"] / days).replace(0, np.nan)
    metric["turnover"] = metric["total_demand"] / average_inventory
    metric["DOI"] = metric["avg_inventory"] / average_daily_demand
    unit_margin = metric["unit_price"] - metric["unit_cost"]
    if (unit_margin < 0).any():
        affected = metric.loc[unit_margin < 0, _SKU_COL].tolist()
        raise ValueError(
            "Stockout opportunity cost requires nonnegative unit margin; "
            f"affected SKUs: {affected}"
        )
    metric["stockout_cost"] = (
        metric["total_lost"] * unit_margin * stockout_cost_multiplier
    )
    metric["holding_cost"] = (
        metric["avg_inventory"]
        * metric["unit_cost"]
        * holding_rate
        * (days / 365)
    )
    metric["ordering_cost"] = metric["num_orders"] * ordering_cost
    metric["total_cost"] = (
        metric["holding_cost"]
        + metric["ordering_cost"]
        + metric["stockout_cost"]
    )
    return metric[
        [
            _SKU_COL,
            "class",
            "total_demand",
            "total_sales",
            "total_lost",
            "avg_inventory",
            "fill_rate",
            "cycle_service_level",
            "cycles_without_stockout",
            "num_cycles",
            "turnover",
            "DOI",
            "stockout_cost",
            "holding_cost",
            "ordering_cost",
            "total_cost",
        ]
    ]


def _service_floor_for_class(
    service_floor: float | Mapping[str, float] | None,
    *,
    class_name: str,
    default_target_percentage: float,
) -> float:
    if service_floor is None:
        value = float(default_target_percentage) / 100
    elif isinstance(service_floor, Mapping):
        if class_name not in service_floor:
            raise ValueError(
                f"No calibration service floor was provided for class {class_name}."
            )
        value = float(service_floor[class_name])
    else:
        value = float(service_floor)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Calibration service floors must be probabilities in [0, 1].")
    return value


def _sku_positive_constraint(
    value: float | Mapping[str, float],
    *,
    sku_id: str,
    label: str,
) -> float:
    """Resolve a scalar or SKU-specific operational constraint."""

    if isinstance(value, Mapping):
        if sku_id not in value:
            raise ValueError(f"No {label} was provided for {sku_id}.")
        resolved = float(value[sku_id])
    else:
        resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{label} for {sku_id} must be finite and positive.")
    return resolved


def optimize_policy_grid(
    validation_forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sku_class_df: pd.DataFrame,
    optimization_scope_df: pd.DataFrame | None = None,
    *,
    q_multipliers: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    safety_stock_quantiles: Iterable[float] = (
        0.50,
        0.75,
        0.85,
        0.90,
        0.95,
        0.97,
        0.99,
    ),
    service_floor: float | Mapping[str, float] | None = None,
    minimum_order_quantity: float | Mapping[str, float] = 1.0,
    order_multiple: float | Mapping[str, float] = 1.0,
    ordering_cost: float | None = None,
    holding_rate: float | None = None,
    stockout_cost_multiplier: float = 1.0,
    require_feasible: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select Q and residual-quantile ROP on calibration data only.

    The final holdout is rejected before candidate construction.  Every
    candidate for a SKU is evaluated on the same validation demand path and
    same pre-calibration inventory state.  Feasible candidates must meet the
    empirical fill-rate floor; total modeled cost is the objective. Candidate Q
    is rounded up to the configured order multiple after enforcing MOQ. The
    returned second dataframe is the complete candidate audit table and can be
    written directly as a CSV artifact.

    ``service_floor=None`` uses the existing class/intervention target as the
    empirical calibration fill-rate floor.  Callers that distinguish a
    lead-time coverage target from a fill-rate SLA should pass an explicit
    probability or class mapping.
    """
    calibration = _validated_calibration_forecasts(validation_forecast_df)
    base_policy = compute_policy_params(
        calibration,
        inventory_df,
        sku_class_df,
        optimization_scope_df,
    )

    q_values = []
    for value in q_multipliers:
        q_values.append(float(value))
    q_grid = sorted(set(q_values))

    invalid_q_value = False
    for value in q_grid:
        if not np.isfinite(value) or value <= 0:
            invalid_q_value = True
    if not q_grid or invalid_q_value:
        raise ValueError("Q multipliers must be a non-empty set of positive values.")

    quantile_values = []
    for value in safety_stock_quantiles:
        quantile_values.append(float(value))
    quantile_grid = sorted(set(quantile_values))

    invalid_quantile = False
    for value in quantile_grid:
        if not np.isfinite(value) or not 0 < value < 1:
            invalid_quantile = True
    if not quantile_grid or invalid_quantile:
        raise ValueError(
            "Safety-stock quantiles must be a non-empty set inside (0, 1)."
        )
    inventory = inventory_df.copy()
    inventory[_DATE_COL] = pd.to_datetime(inventory[_DATE_COL], errors="raise")
    selected_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for _, base in base_policy.iterrows():
        sku_id = base[_SKU_COL]
        class_name = base["class"]
        group = calibration[calibration[_SKU_COL] == sku_id].sort_values(
            _DATE_COL
        )
        if group.empty:
            raise ValueError(f"Calibration rows are missing for {sku_id}.")
        lead_time = _positive_integer_lead_time(
            base["lead_time"], context=f"Grid calibration for {sku_id}"
        )
        cumulative_error = (
            (group["demand"] - group["forecast"])
            .rolling(lead_time, min_periods=lead_time)
            .sum()
            .dropna()
        )
        if cumulative_error.empty:
            raise ValueError(
                f"Grid calibration has insufficient lead-time residuals for {sku_id}."
            )

        calibration_start = group[_DATE_COL].min()
        prior_inventory = inventory[
            (inventory[_SKU_COL] == sku_id)
            & (inventory[_DATE_COL] < calibration_start)
        ].sort_values(_DATE_COL)
        if prior_inventory.empty:
            raise ValueError(
                f"Grid calibration is missing pre-period inventory for {sku_id}."
            )
        initial_inventory = float(prior_inventory["inventory_level"].iloc[-1])
        floor = _service_floor_for_class(
            service_floor,
            class_name=class_name,
            default_target_percentage=base["target_service_level"],
        )
        sku_moq = _sku_positive_constraint(
            minimum_order_quantity,
            sku_id=sku_id,
            label="minimum order quantity",
        )
        sku_order_multiple = _sku_positive_constraint(
            order_multiple,
            sku_id=sku_id,
            label="order multiple",
        )
        # Always include the existing target quantile so the grid can reproduce
        # the backward-compatible policy before varying Q.
        target_quantile = float(base["target_service_level"]) / 100
        sku_quantiles = list(quantile_grid)
        sku_quantiles.append(target_quantile)
        sku_quantiles = sorted(set(sku_quantiles))

        sku_candidate_rows = []
        for q_multiplier in q_grid:
            for quantile in sku_quantiles:
                safety_stock = max(
                    0.0, float(np.quantile(cumulative_error, quantile))
                )

                unconstrained_quantity = (
                    float(base["order_quantity"]) * q_multiplier
                )
                order_quantity = max(unconstrained_quantity, sku_moq)
                order_quantity = (
                    np.ceil(order_quantity / sku_order_multiple)
                    * sku_order_multiple
                )
                rop = (
                    float(group["forecast"].mean()) * lead_time
                    + safety_stock
                )

                candidate_policy = pd.Series(
                    {
                        _SKU_COL: sku_id,
                        "class": class_name,
                        "lead_time": lead_time,
                        "unit_cost": float(base["unit_cost"]),
                        "unit_price": float(base["unit_price"]),
                        "order_quantity": order_quantity,
                        "safety_stock": safety_stock,
                        "rop": rop,
                    }
                )

                # Use the same rule as final evaluation: today's order decision
                # uses the forecast whose target date is tomorrow.
                simulated = _simulate_sku(
                    group[[_DATE_COL, "demand"]],
                    candidate_policy,
                    initial_inventory,
                    forecast_schedule=group[
                        [_SKU_COL, _DATE_COL, "forecast"]
                    ],
                )
                metric = compute_policy_metric(
                    simulated,
                    pd.DataFrame([candidate_policy]),
                    ordering_cost=ordering_cost,
                    holding_rate=holding_rate,
                    stockout_cost_multiplier=stockout_cost_multiplier,
                ).iloc[0]

                candidate = {
                    _SKU_COL: sku_id,
                    "class": class_name,
                    "service_floor": floor,
                    "q_multiplier": q_multiplier,
                    "minimum_order_quantity": sku_moq,
                    "order_multiple": sku_order_multiple,
                    "safety_stock_quantile": quantile,
                    "order_quantity": order_quantity,
                    "safety_stock": safety_stock,
                    "rop": rop,
                    "fill_rate": float(metric["fill_rate"]),
                    "cycle_service_level": float(
                        metric["cycle_service_level"]
                    ),
                    "avg_inventory": float(metric["avg_inventory"]),
                    "stockout_cost": float(metric["stockout_cost"]),
                    "holding_cost": float(metric["holding_cost"]),
                    "ordering_cost": float(metric["ordering_cost"]),
                    "total_cost": float(metric["total_cost"]),
                }
                candidate["service_floor_met"] = bool(
                    np.isfinite(candidate["fill_rate"])
                    and candidate["fill_rate"] >= floor
                )
                sku_candidate_rows.append(candidate)
                candidate_rows.append(candidate)

        candidates = pd.DataFrame(sku_candidate_rows)
        feasible = candidates[candidates["service_floor_met"]]
        if feasible.empty:
            if require_feasible:
                achieved = candidates["fill_rate"].max()
                raise ValueError(
                    f"No feasible policy candidate for {sku_id}: service floor "
                    f"{floor:.4f}, best calibration fill rate {achieved:.4f}."
                )
            pool = candidates.sort_values(
                ["fill_rate", "total_cost", "avg_inventory"],
                ascending=[False, True, True],
            )
            selection_status = "no_feasible_candidate_best_service"
        else:
            pool = feasible.sort_values(
                ["total_cost", "avg_inventory", "fill_rate"],
                ascending=[True, True, False],
            )
            selection_status = "minimum_cost_feasible"
        chosen = pool.iloc[0]
        selected = base.to_dict()
        selected.update(
            {
                "order_quantity": chosen["order_quantity"],
                "safety_stock": chosen["safety_stock"],
                "rop": chosen["rop"],
                "selected_q_multiplier": chosen["q_multiplier"],
                "selected_safety_stock_quantile": chosen[
                    "safety_stock_quantile"
                ],
                "calibration_service_floor": floor,
                "calibration_fill_rate": chosen["fill_rate"],
                "calibration_cycle_service_level": chosen[
                    "cycle_service_level"
                ],
                "calibration_avg_inventory": chosen["avg_inventory"],
                "calibration_total_cost": chosen["total_cost"],
                "calibration_service_floor_met": chosen[
                    "service_floor_met"
                ],
                "selection_status": selection_status,
            }
        )
        selected_rows.append(selected)

    selected_policy = pd.DataFrame(selected_rows)
    candidate_audit = pd.DataFrame(candidate_rows)
    candidate_audit = candidate_audit.sort_values(
        [_SKU_COL, "q_multiplier", "safety_stock_quantile"]
    ).reset_index(drop=True)
    return (
        selected_policy,
        candidate_audit,
    )


def cost_assumption_sensitivity(
    old_sim_results: pd.DataFrame,
    new_sim_results: pd.DataFrame,
    old_policy_df: pd.DataFrame,
    new_policy_df: pd.DataFrame,
    *,
    ordering_costs: Iterable[float] | None = None,
    holding_rates: Iterable[float] | None = None,
    stockout_cost_multipliers: Iterable[float] = (0.5, 1.0, 1.5),
) -> pd.DataFrame:
    """Reprice one paired policy simulation across transparent cost scenarios."""
    keys = [_SKU_COL, _DATE_COL]
    for label, results in [("old", old_sim_results), ("new", new_sim_results)]:
        required = set(keys)
        required.add("demand")
        missing = sorted(required - set(results.columns))
        if missing:
            raise ValueError(f"{label} simulation is missing columns: {missing}")
        if results.duplicated(keys).any():
            raise ValueError(f"{label} simulation contains duplicate SKU-date rows.")
    paired_path = old_sim_results[keys + ["demand"]].merge(
        new_sim_results[keys + ["demand"]],
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired_path["_merge"].eq("both").all() or not np.allclose(
        paired_path["demand_old"],
        paired_path["demand_new"],
        equal_nan=False,
    ):
        raise ValueError(
            "Cost sensitivity requires identical old/new SKU-date demand paths."
        )

    if ordering_costs is None:
        ordering_grid = [
            _ORDER_COST * 0.5,
            _ORDER_COST,
            _ORDER_COST * 1.5,
        ]
    else:
        ordering_grid = []
        for value in ordering_costs:
            ordering_grid.append(float(value))

    if holding_rates is None:
        holding_grid = [
            _HOLD_RATE * 0.5,
            _HOLD_RATE,
            _HOLD_RATE * 1.5,
        ]
    else:
        holding_grid = []
        for value in holding_rates:
            holding_grid.append(float(value))

    stockout_grid = []
    for value in stockout_cost_multipliers:
        stockout_grid.append(float(value))
    for label, values in [
        ("ordering_costs", ordering_grid),
        ("holding_rates", holding_grid),
        ("stockout_cost_multipliers", stockout_grid),
    ]:
        has_invalid_value = False
        for value in values:
            if not np.isfinite(value) or value < 0:
                has_invalid_value = True
        if not values or has_invalid_value:
            raise ValueError(
                f"{label} must be a non-empty set of finite nonnegative values."
            )

    unique_ordering_costs = sorted(set(ordering_grid))
    unique_holding_rates = sorted(set(holding_grid))
    unique_stockout_multipliers = sorted(set(stockout_grid))

    rows = []
    scenario_number = 0
    for ordering_cost in unique_ordering_costs:
        for holding_rate in unique_holding_rates:
            for stockout_multiplier in unique_stockout_multipliers:
                scenario_number += 1

                old_metric = compute_policy_metric(
                    old_sim_results,
                    old_policy_df,
                    ordering_cost=ordering_cost,
                    holding_rate=holding_rate,
                    stockout_cost_multiplier=stockout_multiplier,
                )
                new_metric = compute_policy_metric(
                    new_sim_results,
                    new_policy_df,
                    ordering_cost=ordering_cost,
                    holding_rate=holding_rate,
                    stockout_cost_multiplier=stockout_multiplier,
                )

                old_total_cost = float(old_metric["total_cost"].sum())
                new_total_cost = float(new_metric["total_cost"].sum())
                old_inventory = float(old_metric["avg_inventory"].sum())
                new_inventory = float(new_metric["avg_inventory"].sum())
                old_fill = float(
                    old_metric["total_sales"].sum()
                    / old_metric["total_demand"].sum()
                )
                new_fill = float(
                    new_metric["total_sales"].sum()
                    / new_metric["total_demand"].sum()
                )

                rows.append(
                    {
                        "scenario_id": (
                            f"cost_sensitivity_{scenario_number:03d}"
                        ),
                        "ordering_cost_per_order": ordering_cost,
                        "annual_holding_rate": holding_rate,
                        "stockout_margin_multiplier": stockout_multiplier,
                        "old_fill_rate": old_fill,
                        "new_fill_rate": new_fill,
                        "fill_rate_delta_percentage_points": 100
                        * (new_fill - old_fill),
                        "old_avg_inventory": old_inventory,
                        "new_avg_inventory": new_inventory,
                        "avg_inventory_change_pct": (
                            100 * (new_inventory / old_inventory - 1)
                            if old_inventory > 0
                            else np.nan
                        ),
                        "old_stockout_cost": float(
                            old_metric["stockout_cost"].sum()
                        ),
                        "new_stockout_cost": float(
                            new_metric["stockout_cost"].sum()
                        ),
                        "old_holding_cost": float(
                            old_metric["holding_cost"].sum()
                        ),
                        "new_holding_cost": float(
                            new_metric["holding_cost"].sum()
                        ),
                        "old_ordering_cost": float(
                            old_metric["ordering_cost"].sum()
                        ),
                        "new_ordering_cost": float(
                            new_metric["ordering_cost"].sum()
                        ),
                        "old_total_cost": old_total_cost,
                        "new_total_cost": new_total_cost,
                        "total_cost_savings": (
                            old_total_cost - new_total_cost
                        ),
                        "total_cost_change_pct": (
                            100 * (new_total_cost / old_total_cost - 1)
                            if old_total_cost > 0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _sample_residual_blocks(
    residuals: np.ndarray,
    *,
    horizon_days: int,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contiguous residual blocks and truncate to the requested horizon."""
    if residuals.ndim != 1 or len(residuals) < block_days:
        raise ValueError(
            "Residual block bootstrap requires at least one complete block."
        )
    sampled: list[float] = []
    latest_start = len(residuals) - block_days
    while len(sampled) < horizon_days:
        start = int(rng.integers(0, latest_start + 1))
        sampled.extend(residuals[start : start + block_days])
    return np.asarray(sampled[:horizon_days], dtype=float)


def _build_calibration_residual_history(
    calibration: pd.DataFrame,
    residual_block_days: int,
) -> dict[object, np.ndarray]:
    """Store one ordered calibration-residual history for each SKU."""
    residual_by_sku: dict[object, np.ndarray] = {}
    too_short = []

    for sku_id, group in calibration.groupby(_SKU_COL, observed=True):
        residuals = (group["demand"] - group["forecast"]).to_numpy(
            dtype=float
        )
        residual_by_sku[sku_id] = residuals
        if len(residuals) < residual_block_days:
            too_short.append(sku_id)

    too_short = sorted(too_short)
    if too_short:
        raise ValueError(
            "Calibration residual history is shorter than residual_block_days "
            f"for SKUs: {too_short}"
        )
    return residual_by_sku


def _prepare_final_forecast_schedules(
    final_forecast_df: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, dict[object, pd.DataFrame]]:
    """Validate final forecasts and split them into one schedule per SKU."""
    required_columns = {_SKU_COL, _DATE_COL, "forecast"}
    missing_columns = sorted(
        required_columns - set(final_forecast_df.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Final forecast is missing columns: {missing_columns}"
        )

    # Only copy these three columns. If actual demand is attached to the input,
    # it cannot accidentally be used to create a future scenario.
    final_forecast = final_forecast_df[
        [_SKU_COL, _DATE_COL, "forecast"]
    ].copy()
    final_forecast[_DATE_COL] = pd.to_datetime(
        final_forecast[_DATE_COL], errors="raise"
    )

    starts_before_holdout = (
        final_forecast[_DATE_COL] < _FINAL_TEST_START
    ).any()
    if final_forecast.empty or starts_before_holdout:
        raise ValueError(
            "Final forecasts must start on or after the locked-evaluation boundary."
        )
    if final_forecast.duplicated([_SKU_COL, _DATE_COL]).any():
        raise ValueError("Final forecast contains duplicate SKU-date rows.")

    forecast_values = pd.to_numeric(
        final_forecast["forecast"], errors="coerce"
    )
    invalid_forecast = (
        forecast_values.isna().any()
        or not np.isfinite(forecast_values).all()
        or (forecast_values < 0).any()
    )
    if invalid_forecast:
        raise ValueError(
            "Final forecast values must be finite and nonnegative."
        )
    final_forecast["forecast"] = forecast_values.astype(float)

    common_dates: pd.DatetimeIndex | None = None
    schedule_by_sku: dict[object, pd.DataFrame] = {}
    for sku_id, group in final_forecast.groupby(_SKU_COL, observed=True):
        schedule = _validated_daily_frame(
            group,
            label=f"Final forecast for {sku_id}",
            required_columns=["forecast"],
        )
        sku_dates = pd.DatetimeIndex(schedule[_DATE_COL])
        if common_dates is None:
            common_dates = sku_dates
        elif not sku_dates.equals(common_dates):
            raise ValueError(
                "Final forecast must contain the same calendar for every SKU."
            )
        schedule_by_sku[sku_id] = schedule[
            [_DATE_COL, "forecast"]
        ].copy()

    if common_dates is None or common_dates.empty:
        raise ValueError("Final forecast must contain at least one date.")
    if common_dates[0] != _FINAL_TEST_START:
        raise ValueError(
            "Paired stress scenarios require final forecasts to begin at the "
            "locked-evaluation boundary so initial inventory is leakage-safe."
        )
    return common_dates, schedule_by_sku


def _prepare_stress_test_policies(
    old_policy_df: pd.DataFrame,
    new_policy_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[object], dict[object, int]]:
    """Check that both policies cover the same SKUs and lead times."""
    required_columns = {
        _SKU_COL,
        "class",
        "rop",
        "order_quantity",
        "lead_time",
        "unit_cost",
        "unit_price",
    }

    policies = {}
    policy_inputs = [
        ("old", old_policy_df),
        ("new", new_policy_df),
    ]
    for label, policy in policy_inputs:
        missing_columns = sorted(
            required_columns - set(policy.columns)
        )
        if label == "new" and "safety_stock" not in policy.columns:
            missing_columns.append("safety_stock")
        if missing_columns:
            missing_columns = sorted(set(missing_columns))
            raise ValueError(
                f"{label} policy is missing columns: {missing_columns}"
            )
        if policy.duplicated(_SKU_COL).any():
            raise ValueError(
                f"{label} policy must contain one row per SKU."
            )
        policies[label] = policy.copy().set_index(_SKU_COL, drop=False)

    old_skus = set(policies["old"].index)
    new_skus = set(policies["new"].index)
    if old_skus != new_skus or not old_skus:
        raise ValueError(
            "Old and new policies must contain the same non-empty SKU set."
        )
    target_skus = sorted(old_skus)

    base_lead_time_by_sku = {}
    for sku_id in target_skus:
        old_lead_time = _positive_integer_lead_time(
            policies["old"].loc[sku_id, "lead_time"],
            context=f"Old stress-test policy for {sku_id}",
        )
        new_lead_time = _positive_integer_lead_time(
            policies["new"].loc[sku_id, "lead_time"],
            context=f"New stress-test policy for {sku_id}",
        )
        if old_lead_time != new_lead_time:
            raise ValueError(
                "Paired lead-time stress requires the same base lead time "
                f"for {sku_id}."
            )
        base_lead_time_by_sku[sku_id] = old_lead_time

    return policies, target_skus, base_lead_time_by_sku


def monte_carlo_policy_stress_test(
    calibration_forecast_df: pd.DataFrame,
    final_forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    old_policy_df: pd.DataFrame,
    new_policy_df: pd.DataFrame,
    *,
    n_scenarios: int = 500,
    horizon_days: int | None = None,
    residual_block_days: int = 7,
    lead_time_multipliers: Iterable[float] = (0.8, 1.0, 1.2, 1.5),
    lead_time_probabilities: Iterable[float] | None = None,
    seed: int = 42,
    ordering_cost: float | None = None,
    holding_rate: float | None = None,
    stockout_cost_multiplier: float | None = None,
) -> pd.DataFrame:
    """Run paired forward stress scenarios without final-demand leakage.

    For each SKU and day, future demand is anchored on the corresponding daily
    *forecast* in ``final_forecast_df`` and perturbed with a moving-block
    bootstrap of causal residuals from the dedicated policy-calibration window.
    Actual demand in ``final_forecast_df``, if present, is never selected or
    read. Negative simulated demand is clipped to zero.

    Each scenario draws one portfolio-wide lead-time multiplier and applies the
    same rounded-up shock to both policies. Old and proposed policies also share
    the exact SKU demand paths and pre-holdout initial on-hand state, preserving
    a paired comparison. The historical policy retains its static recorded ROP.
    The proposed policy receives the daily final-forecast schedule, so an
    end-of-day t decision uses forecast t+1 to calculate dynamic ROP under the
    stressed lead time.

    The result has one artifact-ready row per scenario. These rows describe
    sensitivity to the stated residual and lead-time sampling assumptions; they
    are not calibrated prediction intervals, causal savings, or a substitute
    for a new untouched evaluation period.
    """
    if not isinstance(n_scenarios, (int, np.integer)) or n_scenarios <= 0:
        raise ValueError("n_scenarios must be a positive integer.")
    if (
        not isinstance(residual_block_days, (int, np.integer))
        or residual_block_days <= 0
    ):
        raise ValueError("residual_block_days must be a positive integer.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")

    calibration = _validated_calibration_forecasts(
        calibration_forecast_df
    )
    residual_by_sku = _build_calibration_residual_history(
        calibration,
        int(residual_block_days),
    )
    forecast_dates, forecast_schedule_by_sku = (
        _prepare_final_forecast_schedules(final_forecast_df)
    )

    if horizon_days is None:
        horizon_days = len(forecast_dates)
    if not isinstance(horizon_days, (int, np.integer)) or horizon_days <= 1:
        raise ValueError("horizon_days must be an integer greater than one.")
    horizon_days = int(horizon_days)
    if horizon_days > len(forecast_dates):
        raise ValueError(
            f"horizon_days={horizon_days} exceeds the available final forecast "
            f"calendar of {len(forecast_dates)} days."
        )
    scenario_dates = forecast_dates[:horizon_days]

    policies, target_skus, base_lead_time_by_sku = (
        _prepare_stress_test_policies(old_policy_df, new_policy_df)
    )

    for label, available in [
        ("calibration residuals", set(residual_by_sku)),
        ("final forecasts", set(forecast_schedule_by_sku)),
    ]:
        missing = sorted(set(target_skus) - available)
        if missing:
            raise ValueError(f"{label} are missing policy SKUs: {missing}")

    initial = _initial_inventory(inventory_df, pd.Series(target_skus)).set_index(
        _SKU_COL
    )
    missing_initial = sorted(set(target_skus) - set(initial.index))
    if missing_initial:
        raise ValueError(
            f"Stress test is missing pre-holdout initial inventory: {missing_initial}"
        )

    multiplier_values = []
    for value in lead_time_multipliers:
        multiplier_values.append(float(value))
    multipliers = np.asarray(multiplier_values, dtype=float)
    if (
        multipliers.size == 0
        or not np.isfinite(multipliers).all()
        or (multipliers <= 0).any()
    ):
        raise ValueError(
            "lead_time_multipliers must be finite positive values."
        )
    if lead_time_probabilities is None:
        probabilities = np.full(multipliers.size, 1 / multipliers.size)
    else:
        probability_values = []
        for value in lead_time_probabilities:
            probability_values.append(float(value))
        probabilities = np.asarray(probability_values, dtype=float)
        if (
            probabilities.size != multipliers.size
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or probabilities.sum() <= 0
        ):
            raise ValueError(
                "lead_time_probabilities must be finite nonnegative values "
                "matching lead_time_multipliers and summing above zero."
            )
        probabilities = probabilities / probabilities.sum()

    shortage_multiplier = (
        float(_SHORTAGE_COST_MULTIPLIER)
        if stockout_cost_multiplier is None
        else float(stockout_cost_multiplier)
    )
    resolved_ordering_cost = float(
        _ORDER_COST if ordering_cost is None else ordering_cost
    )
    resolved_holding_rate = float(
        _HOLD_RATE if holding_rate is None else holding_rate
    )

    calibration_window_start = str(
        calibration[_DATE_COL].min().date()
    )
    calibration_window_end = str(
        calibration[_DATE_COL].max().date()
    )
    forecast_window_start = str(forecast_dates[0].date())
    forecast_window_end = str(scenario_dates[-1].date())
    scenario_probability = 1 / int(n_scenarios)

    rng = np.random.default_rng(int(seed))
    rows = []
    for scenario_number in range(1, int(n_scenarios) + 1):
        lead_multiplier = float(rng.choice(multipliers, p=probabilities))
        old_simulations = []
        new_simulations = []
        clipped_sku_days = 0
        sampled_residuals = []
        stressed_lead_times = []

        for sku_id in target_skus:
            residual_path = _sample_residual_blocks(
                residual_by_sku[sku_id],
                horizon_days=horizon_days,
                block_days=int(residual_block_days),
                rng=rng,
            )
            sampled_residuals.extend(residual_path)
            daily_forecast_schedule = forecast_schedule_by_sku[sku_id].iloc[
                :horizon_days
            ].reset_index(drop=True)
            raw_demand = (
                daily_forecast_schedule["forecast"].to_numpy(dtype=float)
                + residual_path
            )
            clipped_sku_days += int((raw_demand < 0).sum())
            scenario_demand = pd.DataFrame(
                {
                    _DATE_COL: scenario_dates,
                    "demand": np.clip(raw_demand, 0.0, None),
                }
            )
            stressed_lead_time_value = (
                base_lead_time_by_sku[sku_id] * lead_multiplier
            )
            stressed_lead_time = int(
                np.ceil(stressed_lead_time_value)
            )
            stressed_lead_time = max(1, stressed_lead_time)
            stressed_lead_times.append(stressed_lead_time)

            old_policy = policies["old"].loc[sku_id].copy()
            new_policy = policies["new"].loc[sku_id].copy()
            old_policy["lead_time"] = stressed_lead_time
            new_policy["lead_time"] = stressed_lead_time
            initial_inventory = float(
                initial.loc[sku_id, "initial_inventory"]
            )
            old_simulations.append(
                _simulate_sku(
                    scenario_demand,
                    old_policy,
                    initial_inventory,
                )
            )
            new_simulations.append(
                _simulate_sku(
                    scenario_demand,
                    new_policy,
                    initial_inventory,
                    forecast_schedule=daily_forecast_schedule,
                )
            )

        old_results = pd.concat(old_simulations, ignore_index=True)
        new_results = pd.concat(new_simulations, ignore_index=True)
        old_metric = compute_policy_metric(
            old_results,
            policies["old"].reset_index(drop=True),
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
            stockout_cost_multiplier=shortage_multiplier,
        )
        new_metric = compute_policy_metric(
            new_results,
            policies["new"].reset_index(drop=True),
            ordering_cost=ordering_cost,
            holding_rate=holding_rate,
            stockout_cost_multiplier=shortage_multiplier,
        )

        old_demand = float(old_metric["total_demand"].sum())
        new_demand = float(new_metric["total_demand"].sum())
        if not np.isclose(old_demand, new_demand):
            raise RuntimeError("Paired stress scenario demand paths diverged.")
        old_sales = float(old_metric["total_sales"].sum())
        new_sales = float(new_metric["total_sales"].sum())
        old_fill = old_sales / old_demand if old_demand > 0 else np.nan
        new_fill = new_sales / new_demand if new_demand > 0 else np.nan
        old_inventory = float(old_metric["avg_inventory"].sum())
        new_inventory = float(new_metric["avg_inventory"].sum())
        old_total_cost = float(old_metric["total_cost"].sum())
        new_total_cost = float(new_metric["total_cost"].sum())
        old_cycles = float(old_metric["num_cycles"].sum())
        new_cycles = float(new_metric["num_cycles"].sum())

        old_cycle_service = np.nan
        if old_cycles > 0:
            old_cycle_service = (
                float(old_metric["cycles_without_stockout"].sum())
                / old_cycles
            )
        new_cycle_service = np.nan
        if new_cycles > 0:
            new_cycle_service = (
                float(new_metric["cycles_without_stockout"].sum())
                / new_cycles
            )

        inventory_change_pct = np.nan
        if old_inventory > 0:
            inventory_change_pct = 100 * (
                new_inventory / old_inventory - 1
            )
        total_cost_change_pct = np.nan
        if old_total_cost > 0:
            total_cost_change_pct = 100 * (
                new_total_cost / old_total_cost - 1
            )

        rows.append(
            {
                "scenario_id": f"policy_stress_{scenario_number:05d}",
                "scenario_seed": int(seed),
                "scenario_probability": scenario_probability,
                "horizon_start": str(scenario_dates[0].date()),
                "horizon_end": str(scenario_dates[-1].date()),
                "horizon_days": horizon_days,
                "sku_count": len(target_skus),
                "calibration_window_start": calibration_window_start,
                "calibration_window_end": calibration_window_end,
                "forecast_anchor_window_start": forecast_window_start,
                "forecast_anchor_window_end": forecast_window_end,
                "residual_block_days": int(residual_block_days),
                "lead_time_multiplier": lead_multiplier,
                "ordering_cost_per_order": resolved_ordering_cost,
                "annual_holding_rate": resolved_holding_rate,
                "stockout_margin_multiplier": shortage_multiplier,
                "mean_stressed_lead_time_days": float(
                    np.mean(stressed_lead_times)
                ),
                "sampled_residual_mean": float(
                    np.mean(sampled_residuals)
                ),
                "sampled_residual_std": float(
                    np.std(sampled_residuals, ddof=1)
                ),
                "clipped_negative_demand_sku_days": clipped_sku_days,
                "total_demand": old_demand,
                "old_fill_rate": old_fill,
                "new_fill_rate": new_fill,
                "fill_rate_delta_percentage_points": 100
                * (new_fill - old_fill),
                "old_cycle_service_level": old_cycle_service,
                "new_cycle_service_level": new_cycle_service,
                "old_avg_inventory": old_inventory,
                "new_avg_inventory": new_inventory,
                "avg_inventory_change_pct": inventory_change_pct,
                "old_num_orders": float(old_results["order_placed"].sum()),
                "new_num_orders": float(new_results["order_placed"].sum()),
                "old_stockout_cost": float(
                    old_metric["stockout_cost"].sum()
                ),
                "new_stockout_cost": float(
                    new_metric["stockout_cost"].sum()
                ),
                "old_holding_cost": float(
                    old_metric["holding_cost"].sum()
                ),
                "new_holding_cost": float(
                    new_metric["holding_cost"].sum()
                ),
                "old_ordering_cost": float(
                    old_metric["ordering_cost"].sum()
                ),
                "new_ordering_cost": float(
                    new_metric["ordering_cost"].sum()
                ),
                "old_total_cost": old_total_cost,
                "new_total_cost": new_total_cost,
                "total_cost_savings": old_total_cost - new_total_cost,
                "total_cost_change_pct": total_cost_change_pct,
                "demand_sampling_method": (
                    "sku_calibration_residual_moving_block_bootstrap"
                ),
                "forecast_anchor": "sku_daily_final_forecast_vector",
                "lead_time_sampling_method": (
                    "paired_scenario_level_multiplier_rounded_up"
                ),
                "policy_contract_semantics": (
                    "old_static_rop_new_dynamic_next_day_forecast_rop"
                ),
                "pairing_semantics": (
                    "common_demand_path_initial_inventory_and_lead_time_shock"
                ),
                "evaluation_semantics": (
                    "forward_stress_sensitivity_not_prediction_interval_or_causal_savings"
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_policy_comparison(
    old_policy_metric: pd.DataFrame,
    new_policy_metric: pd.DataFrame,
    n_bootstrap: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate paired-SKU sensitivity intervals for portfolio policy changes.

    Each draw resamples the same SKU positions for both policies, preserving
    the paired comparison. These intervals describe cross-SKU sampling
    sensitivity; they are not future-demand or lead-time prediction intervals.
    """
    required = [
        _SKU_COL,
        "total_demand",
        "total_sales",
        "avg_inventory",
        "total_cost",
    ]
    for label, metric in [("old", old_policy_metric), ("new", new_policy_metric)]:
        missing = []
        for column in required:
            if column not in metric.columns:
                missing.append(column)
        if missing:
            raise ValueError(f"{label} policy metric is missing columns: {missing}")

    paired = old_policy_metric[required].merge(
        new_policy_metric[required],
        on=_SKU_COL,
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if len(paired) != len(old_policy_metric) or len(paired) != len(new_policy_metric):
        raise ValueError("Old and new policy metrics must contain the same SKU set.")
    if paired.empty:
        raise ValueError("Policy comparison requires at least one SKU.")

    def _changes(frame: pd.DataFrame) -> np.ndarray:
        old_demand = frame["total_demand_old"].sum()
        new_demand = frame["total_demand_new"].sum()
        old_fill = frame["total_sales_old"].sum() / old_demand
        new_fill = frame["total_sales_new"].sum() / new_demand
        old_cost = frame["total_cost_old"].sum()
        old_inventory = frame["avg_inventory_old"].sum()
        return np.array(
            [
                100 * (new_fill - old_fill),
                100 * (frame["avg_inventory_new"].sum() / old_inventory - 1),
                100 * (frame["total_cost_new"].sum() / old_cost - 1),
            ]
        )

    if (
        (paired[["total_demand_old", "total_demand_new"]].sum() <= 0).any()
        or paired["avg_inventory_old"].sum() <= 0
        or paired["total_cost_old"].sum() <= 0
    ):
        raise ValueError("Bootstrap comparison requires positive portfolio denominators.")

    point_estimate = _changes(paired)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 3), dtype=float)
    for draw in range(n_bootstrap):
        sampled = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        draws[draw] = _changes(sampled)

    return pd.DataFrame(
        {
            "metric": [
                "fill_rate_delta_percentage_points",
                "avg_inventory_change_pct",
                "total_cost_change_pct",
            ],
            "estimate": point_estimate,
            "ci_2_5": np.quantile(draws, 0.025, axis=0),
            "ci_97_5": np.quantile(draws, 0.975, axis=0),
            "n_bootstrap": n_bootstrap,
            "method": "paired_sku_bootstrap",
        }
    )


def save_simulation_results(
    policy_sku: pd.DataFrame,
    old_policy_metric: pd.DataFrame,
    new_policy_metric: pd.DataFrame,
) -> None:
    os.makedirs(_PRO_DIR, exist_ok=True)
    policy_sku.to_csv(os.path.join(_PRO_DIR, "policy_sku.csv"), index=False)
    old_policy_metric.to_csv(os.path.join(_PRO_DIR, "old_policy_metric.csv"), index=False)
    new_policy_metric.to_csv(os.path.join(_PRO_DIR, "new_policy_metric.csv"), index=False)