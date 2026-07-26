"""Run the project from raw data to dashboard-ready files.

The ``main`` function follows the same order used in the business story:

1. classify SKUs;
2. select and run forecasts;
3. calibrate inventory policies;
4. compare policies;
5. save results and lineage.

Each detailed calculation lives in its own module.  This file only coordinates
those steps.
"""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from data.loader import DataLoader, MANIFEST_REQUIRED_ARTIFACTS
from models.classification import (
    classify_abc,
    classify_xyz,
    compute_sku_metric,
    merge_abc_xyz,
    select_optimization_scope,
    save_optimization_scope,
    save_sku_class,
    save_sku_metric,
)
from models.forecasting import (
    calibration_forecast_baseline_class,
    combine_and_save,
    compare_strategic_candidates,
    evaluate_models,
    evaluate_forecast_hierarchy,
    forecast_ax_ay,
    forecast_baseline_class,
    save_forecast_metrics,
    validation_forecast_ax_ay,
    validation_forecast_baseline_class,
)
from models.simulation import (
    bootstrap_policy_comparison,
    cost_assumption_sensitivity,
    compute_policy_metric,
    estimate_historical_policy,
    estimate_inferred_historical_policy_sensitivity,
    monte_carlo_policy_stress_test,
    optimize_policy_grid,
    save_simulation_results,
    simulate_policy,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _config = json.load(f)

_FINAL_TEST_START = _config["data"]["final_test_start"]
_POLICY_CALIBRATION_START = _config["data"].get(
    "policy_calibration_start", _config["data"]["validation_start"]
)
_POLICY_CALIBRATION_END = _config["data"].get(
    "policy_calibration_end", _config["data"]["validation_end"]
)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PROCESSED_DIR = os.path.join(_ROOT, _config["paths"]["data_processed"])
_METADATA_DIR = os.path.join(_ROOT, "data", "metadata")


def _log_forecast_metrics(forecast_df):
    overall = evaluate_models(
        forecast_df["demand"], {"portfolio": forecast_df["forecast"]}
    )
    logger.info("Locked-evaluation portfolio forecast metrics:\n%s", overall.to_string())
    for class_name, group in forecast_df.groupby("class", observed=True):
        metrics = evaluate_models(group["demand"], {class_name: group["forecast"]})
        logger.info("Locked-evaluation %s metrics:\n%s", class_name, metrics.to_string())


def main():
    logger.info(
        "Pipeline protocol: rolling one-step; locked evaluation starts %s",
        _FINAL_TEST_START,
    )

    loader = DataLoader(mode="pipeline")
    inventory_df = loader.load().inventory

    # Step 1 — Classification is frozen before the locked evaluation window.
    abc_df = classify_abc(inventory_df)
    xyz_df, xyz_comparison, xyz_model = classify_xyz(
        inventory_df, return_diagnostics=True
    )
    sku_class = merge_abc_xyz(abc_df, xyz_df)
    sku_metric = compute_sku_metric(inventory_df, observed_until=_FINAL_TEST_START)
    optimization_scope, class_summary = select_optimization_scope(inventory_df, sku_class)
    save_sku_class(sku_class)
    save_sku_metric(sku_metric)
    save_optimization_scope(optimization_scope)
    df_with_class = inventory_df.merge(sku_class, on="sku_id", how="inner", validate="many_to_one")
    strategic_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "protect_strategic_value", "class"
    ].tolist()
    understock_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "correct_understock", "class"
    ].tolist()
    overstock_classes = optimization_scope.loc[
        optimization_scope["intervention"] == "reduce_overstock", "class"
    ].tolist()
    logger.info("Optimization scope selected from business metrics:\n%s", optimization_scope[["class", "intervention", "fill_rate", "DOI"]].to_string(index=False))
    logger.info(
        "XYZ model-selection window selected %s:\n%s",
        xyz_model,
        xyz_comparison.to_string(index=False),
    )

    # Step 2 — Keep model selection and policy calibration on disjoint dates.
    #
    # Reusing the rows that selected/tuned a model to estimate its safety-stock
    # residual distribution can make that distribution too optimistic.  The
    # forecasting module therefore freezes the method on the selection window,
    # then scores it once on the later calibration window.
    logger.info(
        "Building disjoint model-selection and policy-calibration forecasts."
    )
    lgbm_selection, lgbm_calibration, lgbm_params = validation_forecast_ax_ay(
        df_with_class, strategic_classes
    )
    strategic_comparison, strategic_model, _ = (
        compare_strategic_candidates(
            lgbm_selection, df_with_class, strategic_classes
        )
    )
    strategic_calibration = (
        lgbm_calibration
        if strategic_model == "LightGBM"
        else calibration_forecast_baseline_class(
            df_with_class, strategic_classes, strategic_model
        )
    )
    logger.info(
        "Strategic model-selection window selected %s:\n%s",
        strategic_model,
        strategic_comparison.to_string(index=False),
    )
    _, cx_calibration, cx_model = validation_forecast_baseline_class(
        df_with_class, understock_classes
    )
    _, bz_cz_calibration, bz_cz_model = validation_forecast_baseline_class(
        df_with_class, overstock_classes
    )
    calibration_forecasts = combine_validation(
        strategic_calibration, cx_calibration, bz_cz_calibration
    )

    # Step 3 — Final forecasts happen once after all choices are frozen.
    logger.info("Producing locked-evaluation forecasts.")
    if strategic_model == "LightGBM":
        strategic_final = forecast_ax_ay(
            df_with_class, params=lgbm_params, classes=strategic_classes
        )
    else:
        strategic_final = forecast_baseline_class(
            df_with_class, strategic_classes, selected=strategic_model
        )
    cx_final = forecast_baseline_class(df_with_class, understock_classes, selected=cx_model)
    bz_cz_final = forecast_baseline_class(
        df_with_class, overstock_classes, selected=bz_cz_model
    )
    forecast_df = combine_and_save(strategic_final, cx_final, bz_cz_final)
    forecast_df = forecast_df.merge(sku_class, on="sku_id", how="left", validate="many_to_one")
    _log_forecast_metrics(forecast_df)
    forecast_metrics = evaluate_forecast_hierarchy(
        forecast_df.drop(columns="class"), inventory_df, sku_class
    )
    save_forecast_metrics(forecast_metrics)
    logger.info(
        "Final evaluation hierarchy:\n%s",
        forecast_metrics[forecast_metrics["level"] != "sku"].to_string(index=False),
    )

    # Step 4 — Calibrate on validation, then compare policies with one simulator
    # over the locked evaluation window only.
    inventory_config = _config["inventory"]
    fill_floor_by_intervention = inventory_config[
        "fill_rate_floor_by_intervention"
    ]
    service_floor_by_class = {}
    for _, scope_row in optimization_scope.iterrows():
        class_name = scope_row["class"]
        intervention = scope_row["intervention"]
        service_floor_by_class[class_name] = fill_floor_by_intervention[
            intervention
        ]
    metric_cost_kwargs = {
        "ordering_cost": inventory_config["ordering_cost"],
        "holding_rate": inventory_config["holding_rate"],
        "stockout_cost_multiplier": inventory_config[
            "shortage_cost_multiplier"
        ],
    }

    # Optimize both reorder threshold (s) and order quantity (Q) using only the
    # policy-calibration period.  Optional SKU-level MOQ/order-multiple columns
    # override the documented scalar defaults when they exist in the input.
    new_policy, policy_candidate_audit = optimize_policy_grid(
        calibration_forecasts,
        inventory_df,
        sku_class,
        optimization_scope,
        q_multipliers=inventory_config["q_candidate_multipliers"],
        safety_stock_quantiles=inventory_config["rop_candidate_quantiles"],
        service_floor=service_floor_by_class,
        minimum_order_quantity=_sku_constraint_argument(
            inventory_df,
            "moq",
            inventory_config["minimum_order_quantity_default"],
        ),
        order_multiple=_sku_constraint_argument(
            inventory_df,
            "order_multiple",
            inventory_config["order_multiple_default"],
        ),
        **metric_cost_kwargs,
        # Keep the audit complete when the configured grid cannot meet an SLA.
        # Such SKUs are explicitly marked for review instead of disappearing.
        require_feasible=False,
    )
    policy_candidate_audit.to_csv(
        os.path.join(_PROCESSED_DIR, "policy_candidate_audit.csv"), index=False
    )

    target_skus = new_policy["sku_id"]
    old_policy = estimate_historical_policy(inventory_df, sku_class, target_skus)
    inferred_old_policy = estimate_inferred_historical_policy_sensitivity(
        inventory_df, sku_class, target_skus
    )
    old_results = simulate_policy(inventory_df, old_policy)
    inferred_old_results = simulate_policy(inventory_df, inferred_old_policy)
    new_results = simulate_policy(
        inventory_df,
        new_policy,
        forecast_schedule=forecast_df[["sku_id", "date", "forecast"]],
    )
    old_metric = compute_policy_metric(
        old_results, old_policy, **metric_cost_kwargs
    )
    inferred_old_metric = compute_policy_metric(
        inferred_old_results, inferred_old_policy, **metric_cost_kwargs
    )
    new_metric = compute_policy_metric(
        new_results, new_policy, **metric_cost_kwargs
    )
    save_simulation_results(new_policy, old_metric, new_metric)

    cost_grid = inventory_config["cost_sensitivity"]
    policy_sensitivity = cost_assumption_sensitivity(
        old_results,
        new_results,
        old_policy,
        new_policy,
        ordering_costs=cost_grid["ordering_costs"],
        holding_rates=cost_grid["holding_rates"],
        stockout_cost_multipliers=cost_grid["shortage_cost_multipliers"],
    )
    policy_sensitivity.to_csv(
        os.path.join(_PROCESSED_DIR, "policy_sensitivity.csv"), index=False
    )
    historical_policy_sensitivity = _historical_policy_sensitivity(
        old_policy,
        old_metric,
        inferred_old_policy,
        inferred_old_metric,
    )
    historical_policy_sensitivity.to_csv(
        os.path.join(_PROCESSED_DIR, "historical_policy_sensitivity.csv"),
        index=False,
    )
    policy_action = _build_policy_action(
        inventory_df,
        optimization_scope,
        old_policy,
        new_policy,
        old_metric,
        new_metric,
    )
    policy_action.to_csv(
        os.path.join(_PROCESSED_DIR, "policy_action.csv"), index=False
    )

    # Report both the modeled intervention scope and all 150 SKUs.  Out-of-scope
    # SKUs keep their recorded static policy; in-scope SKUs use the proposed
    # daily forecast-aligned ROP policy.
    all_skus = sku_class["sku_id"]
    old_full_policy = estimate_historical_policy(
        inventory_df, sku_class, all_skus
    )
    in_scope_skus = new_policy["sku_id"]
    out_of_scope_mask = ~old_full_policy["sku_id"].isin(in_scope_skus)
    out_of_scope_policy = old_full_policy[out_of_scope_mask]
    proposed_full_policy = pd.concat(
        [new_policy, out_of_scope_policy],
        ignore_index=True,
        sort=False,
    )
    old_full_results = simulate_policy(inventory_df, old_full_policy)
    proposed_full_results = simulate_policy(
        inventory_df,
        proposed_full_policy,
        forecast_schedule=forecast_df[["sku_id", "date", "forecast"]],
    )
    old_full_metric = compute_policy_metric(
        old_full_results, old_full_policy, **metric_cost_kwargs
    )
    proposed_full_metric = compute_policy_metric(
        proposed_full_results, proposed_full_policy, **metric_cost_kwargs
    )
    full_policy_summary = pd.DataFrame(
        [
            _portfolio_policy_row(
                "recorded_policy_all_skus", old_full_metric
            ),
            _portfolio_policy_row(
                "proposed_in_scope_recorded_out_of_scope",
                proposed_full_metric,
            ),
        ]
    )
    full_policy_summary.to_csv(
        os.path.join(_PROCESSED_DIR, "full_policy_summary.csv"), index=False
    )

    scenario_config = inventory_config["scenario_simulation"]
    scenario_uncertainty = monte_carlo_policy_stress_test(
        calibration_forecasts,
        forecast_df[["sku_id", "date", "forecast"]],
        inventory_df,
        old_policy,
        new_policy,
        n_scenarios=scenario_config["n_scenarios"],
        residual_block_days=scenario_config["residual_block_days"],
        lead_time_multipliers=scenario_config["lead_time_multipliers"],
        lead_time_probabilities=scenario_config["lead_time_probabilities"],
        seed=scenario_config["seed"],
        **metric_cost_kwargs,
    )
    scenario_uncertainty.to_csv(
        os.path.join(_PROCESSED_DIR, "scenario_uncertainty.csv"), index=False
    )

    policy_uncertainty = bootstrap_policy_comparison(old_metric, new_metric)
    policy_uncertainty.to_csv(
        os.path.join(_PROCESSED_DIR, "policy_uncertainty.csv"), index=False
    )

    comparison = _aggregate_policy_by_class(old_metric, "old_").join(
        _aggregate_policy_by_class(new_metric, "new_")
    )
    logger.info("Locked-evaluation simulation comparison:\n%s", comparison.to_string())
    logger.info(
        "Paired-SKU bootstrap sensitivity (95%% interval):\n%s",
        policy_uncertainty.to_string(index=False),
    )
    save_artifact_manifest(
        xyz_model=xyz_model,
        strategic_model=strategic_model,
        understock_model=cx_model,
        overstock_model=bz_cz_model,
        optimization_scope=optimization_scope,
        forecast_metrics=forecast_metrics,
        old_metric=old_metric,
        new_metric=new_metric,
        policy_uncertainty=policy_uncertainty,
        policy_sensitivity=policy_sensitivity,
        full_policy_summary=full_policy_summary,
        scenario_uncertainty=scenario_uncertainty,
    )


def combine_validation(*forecast_frames):
    """Combine and persist the dedicated policy-calibration forecasts.

    The filename remains ``validation_forecast.csv`` for backward compatibility,
    but its rows are now restricted to the later, disjoint calibration window.
    The artifact manifest records the exact window so downstream consumers do
    not have to infer its semantics from the legacy filename.
    """
    validation = pd.concat(forecast_frames, ignore_index=True)
    if validation.duplicated(["sku_id", "date"]).any():
        raise ValueError("Validation forecast contains duplicate SKU-date rows.")
    calibration_start = pd.Timestamp(_POLICY_CALIBRATION_START)
    calibration_end = pd.Timestamp(_POLICY_CALIBRATION_END)
    if (
        (validation["date"] < calibration_start).any()
        or (validation["date"] > calibration_end).any()
        or (validation["date"] >= _FINAL_TEST_START).any()
    ):
        raise ValueError(
            "Policy-calibration forecasts must stay inside the configured "
            f"{calibration_start.date()} to {calibration_end.date()} window."
        )
    os.makedirs(_PROCESSED_DIR, exist_ok=True)
    validation.to_csv(os.path.join(_PROCESSED_DIR, "validation_forecast.csv"), index=False)
    return validation


def _sku_constraint_argument(inventory_df, column, default_value):
    """Return a scalar default or a validated per-SKU operational constraint.

    The synthetic source currently uses portfolio-wide defaults.  This helper
    lets a real extract add ``moq`` or ``order_multiple`` without changing the
    optimization call: the median pre-holdout master-data value is then used
    for each SKU.
    """
    if column not in inventory_df.columns:
        return float(default_value)
    history = inventory_df[
        pd.to_datetime(inventory_df["date"]) < pd.Timestamp(_FINAL_TEST_START)
    ]
    constraints = history.groupby("sku_id", observed=True)[column].median()
    constraints = constraints.astype(float)
    invalid = constraints.isna() | (constraints <= 0)
    if invalid.any():
        affected = constraints.index[invalid].tolist()
        raise ValueError(
            f"{column} must be positive for every SKU; affected: {affected}"
        )
    return constraints.to_dict()


def _portfolio_policy_row(policy_label, metric):
    """Aggregate additive SKU metrics and demand/cycle-weighted rates."""
    total_demand = float(metric["total_demand"].sum())
    total_sales = float(metric["total_sales"].sum())
    cycles = float(metric["num_cycles"].sum())
    cycles_without_stockout = float(metric["cycles_without_stockout"].sum())
    return {
        "policy_label": policy_label,
        "sku_count": int(metric["sku_id"].nunique()),
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_lost": float(metric["total_lost"].sum()),
        "fill_rate": total_sales / total_demand if total_demand else None,
        "cycle_service_level": (
            cycles_without_stockout / cycles if cycles else None
        ),
        "sum_sku_avg_inventory": float(metric["avg_inventory"].sum()),
        "stockout_cost": float(metric["stockout_cost"].sum()),
        "holding_cost": float(metric["holding_cost"].sum()),
        "ordering_cost": float(metric["ordering_cost"].sum()),
        "total_cost": float(metric["total_cost"].sum()),
    }


def _historical_policy_sensitivity(
    recorded_policy,
    recorded_metric,
    inferred_policy,
    inferred_metric,
):
    """Expose the receipt-inferred baseline as sensitivity, not ground truth."""
    rows = []
    for source, policy, metric in [
        ("recorded_reorder_point", recorded_policy, recorded_metric),
        ("receipt_inferred_proxy", inferred_policy, inferred_metric),
    ]:
        row = _portfolio_policy_row(source, metric)
        row["baseline_source"] = source
        row["mean_rop"] = float(policy["rop"].mean())
        row["median_rop"] = float(policy["rop"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_policy_action(
    inventory_df,
    optimization_scope,
    old_policy,
    new_policy,
    old_metric,
    new_metric,
):
    """Build an auditable SKU action list from policy parameters and outcomes."""
    history = inventory_df[
        pd.to_datetime(inventory_df["date"]) < pd.Timestamp(_FINAL_TEST_START)
    ]
    recorded_safety_stock = (
        history[history["sku_id"].isin(new_policy["sku_id"])]
        .groupby("sku_id", observed=True)["safety_stock"]
        .median()
        .rename("current_safety_stock")
        .reset_index()
    )
    current = old_policy[
        ["sku_id", "class", "rop", "order_quantity", "unit_cost"]
    ].rename(
        columns={
            "rop": "current_rop",
            "order_quantity": "current_order_quantity",
        }
    )
    proposed_columns = [
        "sku_id",
        "rop",
        "safety_stock",
        "order_quantity",
        "target_service_level",
        "calibration_service_floor",
        "calibration_fill_rate",
        "calibration_service_floor_met",
        "selection_status",
    ]
    proposed = new_policy[proposed_columns].rename(
        columns={
            "rop": "proposed_rop",
            "safety_stock": "proposed_safety_stock",
            "order_quantity": "proposed_order_quantity",
        }
    )
    old_outcomes = old_metric[
        ["sku_id", "fill_rate", "avg_inventory", "total_cost"]
    ].rename(
        columns={
            "fill_rate": "current_fill_rate",
            "avg_inventory": "current_avg_inventory",
            "total_cost": "current_total_cost",
        }
    )
    new_outcomes = new_metric[
        ["sku_id", "fill_rate", "avg_inventory", "total_cost"]
    ].rename(
        columns={
            "fill_rate": "proposed_fill_rate",
            "avg_inventory": "proposed_avg_inventory",
            "total_cost": "proposed_total_cost",
        }
    )
    action = current.merge(
        recorded_safety_stock,
        on="sku_id",
        validate="one_to_one",
    )
    action = action.merge(proposed, on="sku_id", validate="one_to_one")
    action = action.merge(old_outcomes, on="sku_id", validate="one_to_one")
    action = action.merge(new_outcomes, on="sku_id", validate="one_to_one")

    intervention_by_class = optimization_scope[
        ["class", "intervention"]
    ].drop_duplicates("class")
    action = action.merge(
        intervention_by_class,
        on="class",
        how="left",
        validate="many_to_one",
    )
    action["rop_delta_units"] = action["proposed_rop"] - action["current_rop"]
    action["safety_stock_delta_units"] = (
        action["proposed_safety_stock"] - action["current_safety_stock"]
    )
    action["order_quantity_delta_units"] = (
        action["proposed_order_quantity"] - action["current_order_quantity"]
    )
    action["fill_delta_pp"] = 100 * (
        action["proposed_fill_rate"] - action["current_fill_rate"]
    )
    action["avg_on_hand_delta_units"] = (
        action["proposed_avg_inventory"] - action["current_avg_inventory"]
    )
    action["avg_on_hand_value_delta"] = (
        action["avg_on_hand_delta_units"] * action["unit_cost"]
    )
    action["modeled_cost_saving"] = (
        action["current_total_cost"] - action["proposed_total_cost"]
    )
    action["policy_mode"] = "daily_dynamic_rop_next_day_forecast"

    def recommended_action(row):
        if row["selection_status"] != "minimum_cost_feasible":
            return "Review SLA/grid feasibility"
        changes = []
        if abs(row["rop_delta_units"]) > max(1.0, 0.05 * row["current_rop"]):
            changes.append("raise ROP" if row["rop_delta_units"] > 0 else "lower ROP")
        if abs(row["order_quantity_delta_units"]) > max(
            1.0, 0.05 * row["current_order_quantity"]
        ):
            changes.append(
                "raise Q" if row["order_quantity_delta_units"] > 0 else "lower Q"
            )
        return " + ".join(changes) if changes else "Keep parameters / monitor"

    action["recommended_action"] = action.apply(recommended_action, axis=1)
    action = action.sort_values(
        ["modeled_cost_saving", "fill_delta_pp"],
        ascending=[False, False],
    ).reset_index(drop=True)
    action["priority_rank"] = action.index + 1
    return action


def _aggregate_policy_by_class(metric, prefix):
    """Demand/cycle-weight rates and sum inventory/cost at class level."""
    grouped = metric.groupby("class", observed=True).agg(
        total_sales=("total_sales", "sum"),
        total_demand=("total_demand", "sum"),
        cycles_without_stockout=("cycles_without_stockout", "sum"),
        num_cycles=("num_cycles", "sum"),
        avg_inventory=("avg_inventory", "sum"),
        total_cost=("total_cost", "sum"),
    )
    grouped["fill_rate"] = grouped["total_sales"] / grouped["total_demand"]
    grouped["cycle_service_level"] = (
        grouped["cycles_without_stockout"] / grouped["num_cycles"]
    )
    return grouped[
        ["fill_rate", "cycle_service_level", "avg_inventory", "total_cost"]
    ].add_prefix(prefix)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance():
    """Return honest revision metadata, including uncommitted/untracked work."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "git_commit": commit,
            "working_tree_dirty": bool(status.strip()),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "working_tree_dirty": None,
            "note": "Git metadata unavailable in this runtime.",
        }


def _installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_artifact_manifest(
    *,
    xyz_model,
    strategic_model,
    understock_model,
    overstock_model,
    optimization_scope,
    forecast_metrics,
    old_metric,
    new_metric,
    policy_uncertainty,
    policy_sensitivity=None,
    full_policy_summary=None,
    scenario_uncertainty=None,
):
    """Bind data, assumptions, source lineage, and outputs to one pipeline run."""
    resolved_files = {}
    for relative_path in MANIFEST_REQUIRED_ARTIFACTS:
        path_parts = relative_path.split("/")
        resolved_files[relative_path] = os.path.join(_ROOT, *path_parts)

    missing = []
    for relative_path, absolute_path in resolved_files.items():
        if not os.path.isfile(absolute_path):
            missing.append(relative_path)
    if missing:
        raise FileNotFoundError(
            "Cannot build artifact manifest; missing required artifacts: "
            + ", ".join(missing)
        )

    artifacts = {}
    for relative, absolute in resolved_files.items():
        artifacts[relative] = {
            "sha256": _sha256(absolute),
            "bytes": os.path.getsize(absolute),
        }
    portfolio = forecast_metrics[
        (forecast_metrics["level"] == "portfolio")
        & (forecast_metrics["segment"] == "portfolio")
    ].iloc[0]
    history = pd.read_csv(
        os.path.join(_ROOT, "data", "processed", "inventory_processed.csv"),
        parse_dates=["date"],
    )
    history = history[history["date"] < pd.Timestamp(_FINAL_TEST_START)]
    in_scope_skus = set(new_metric["sku_id"])
    scoped_history = history[history["sku_id"].isin(in_scope_skus)]
    gross_margin_value = (
        history["demand"] * (history["unit_price"] - history["unit_cost"])
    )
    scoped_gross_margin_value = (
        scoped_history["demand"]
        * (scoped_history["unit_price"] - scoped_history["unit_cost"])
    )
    source_paths = [
        "src/backend/main.py",
        "src/backend/data/loader.py",
        "src/backend/features/feature_engineering.py",
        "src/backend/models/classification.py",
        "src/backend/models/forecasting.py",
        "src/backend/models/simulation.py",
    ]
    source_lineage = {}
    for relative_path in source_paths:
        absolute_path = os.path.join(_ROOT, *relative_path.split("/"))
        source_lineage[relative_path] = _sha256(absolute_path)
    sensitivity_summary = None
    if policy_sensitivity is not None and not policy_sensitivity.empty:
        savings = policy_sensitivity["total_cost_savings"]
        sensitivity_summary = {
            "scenario_count": int(len(policy_sensitivity)),
            "positive_savings_scenario_share": float((savings > 0).mean()),
            "minimum_total_cost_savings": float(savings.min()),
            "maximum_total_cost_savings": float(savings.max()),
        }
    scenario_summary = None
    if scenario_uncertainty is not None and not scenario_uncertainty.empty:
        scenario_metrics = [
            "fill_rate_delta_percentage_points",
            "avg_inventory_change_pct",
            "total_cost_change_pct",
            "total_cost_savings",
        ]
        scenario_quantiles = {}
        for metric_name in scenario_metrics:
            metric_values = scenario_uncertainty[metric_name]
            scenario_quantiles[metric_name] = {
                "p05": float(metric_values.quantile(0.05)),
                "p50": float(metric_values.quantile(0.50)),
                "p95": float(metric_values.quantile(0.95)),
            }

        scenario_summary = {
            "scenario_count": int(len(scenario_uncertainty)),
            "method": "paired_residual_block_bootstrap_with_lead_time_shocks",
            "positive_cost_savings_scenario_share": float(
                (scenario_uncertainty["total_cost_savings"] > 0).mean()
            ),
            "quantiles": scenario_quantiles,
        }

    forecast_portfolio = {"scope": "optimization_scope_only"}
    forecast_metric_names = ["MAE", "RMSE", "WAPE", "Bias", "MASE", "RMSSE"]
    for metric_name in forecast_metric_names:
        forecast_portfolio[metric_name] = float(portfolio[metric_name])

    cost_assumptions = {}
    cost_assumption_names = [
        "ordering_cost",
        "holding_rate",
        "shortage_cost_method",
        "shortage_cost_multiplier",
    ]
    for assumption_name in cost_assumption_names:
        cost_assumptions[assumption_name] = _config["inventory"][
            assumption_name
        ]

    full_portfolio_records = None
    if full_policy_summary is not None:
        full_portfolio_records = full_policy_summary.to_dict("records")

    manifest = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "status": "validated_pipeline_run",
        "evaluation_semantics": (
            "locked retrospective evaluation window; excluded from model selection"
        ),
        "protocol": _config["backtest"],
        "data_windows": _config["data"],
        "provenance": {
            **_git_provenance(),
            "source_sha256": source_lineage,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": _installed_version("numpy"),
                "lightgbm": _installed_version("lightgbm"),
                "scipy": _installed_version("scipy"),
            },
        },
        "model_selection": {
            "xyz": xyz_model,
            "strategic": strategic_model,
            "understock": understock_model,
            "overstock": overstock_model,
        },
        "optimization_scope": {
            "segments": optimization_scope[
                ["class", "intervention"]
            ].to_dict("records"),
            "sku_count": int(len(in_scope_skus)),
            "portfolio_sku_count": int(history["sku_id"].nunique()),
            "sku_coverage": float(
                len(in_scope_skus) / history["sku_id"].nunique()
            ),
            "demand_coverage": float(
                scoped_history["demand"].sum() / history["demand"].sum()
            ),
            "economic_value_basis": _config["abc_xyz"]["abc_value_basis"],
            "gross_margin_value_coverage": float(
                scoped_gross_margin_value.sum() / gross_margin_value.sum()
            ),
        },
        "forecast_portfolio": forecast_portfolio,
        "simulation_portfolio": {
            "scope": "optimization_scope_only",
            "historical_policy_source": _config["inventory"][
                "historical_policy_source"
            ],
            "proposed_policy_mode": _config["inventory"][
                "dynamic_rop_forecast_alignment"
            ],
            "cost_assumptions": cost_assumptions,
            "old_total_cost": float(old_metric["total_cost"].sum()),
            "new_total_cost": float(new_metric["total_cost"].sum()),
            "old_fill_rate": float(
                old_metric["total_sales"].sum() / old_metric["total_demand"].sum()
            ),
            "new_fill_rate": float(
                new_metric["total_sales"].sum() / new_metric["total_demand"].sum()
            ),
            "paired_sku_bootstrap": policy_uncertainty.to_dict("records"),
            "cost_sensitivity": sensitivity_summary,
            "future_demand_lead_time_stress": scenario_summary,
        },
        "full_portfolio_simulation": full_portfolio_records,
        "artifacts": artifacts,
    }
    os.makedirs(_METADATA_DIR, exist_ok=True)
    with open(
        os.path.join(_METADATA_DIR, "artifact_manifest.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()