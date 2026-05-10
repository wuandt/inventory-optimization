import logging
import warnings
import os
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from data.loader import DataLoader
from features.feature_engineering import create_features
from models.classification import (
    classify_abc,
    classify_xyz,
    compute_sku_metric,
    merge_abc_xyz,
    save_sku_class,
    save_sku_metric,
)
from models.forecasting import (
    combine_and_save,
    evaluate_models,
    forecast_ax_ay,
    forecast_bz_cz,
    forecast_cx,
)
from models.simulation import (
    compute_new_policy_metric,
    compute_old_policy_metric,
    compute_policy_params,
    save_simulation_results,
    simulate_new_policy,
)


def main():
    logger.info("=" * 50)
    logger.info("STEP 1 — Load & preprocess raw data")
    logger.info("=" * 50)

    loader = DataLoader(mode="pipeline")
    loader.load()
    inventory_df = loader.inventory

    logger.info(f"Inventory shape     : {inventory_df.shape}")
    logger.info(f"SKUs                : {inventory_df['sku_id'].nunique()}")
    logger.info(f"Date range          : {inventory_df['date'].min()} → {inventory_df['date'].max()}")

    logger.info("=" * 50)
    logger.info("STEP 2 — ABC-XYZ classification")
    logger.info("=" * 50)

    abc_df     = classify_abc(inventory_df)
    xyz_df     = classify_xyz(inventory_df)
    sku_class  = merge_abc_xyz(abc_df, xyz_df)
    sku_metric = compute_sku_metric(inventory_df)

    save_sku_class(sku_class)
    save_sku_metric(sku_metric)

    logger.info("ABC-XYZ distribution:")
    logger.info(f"\n{sku_class['class'].value_counts().to_string()}")

    logger.info("=" * 50)
    logger.info("STEP 3 — Demand forecasting")
    logger.info("=" * 50)

    df_with_class = inventory_df.merge(sku_class, on="sku_id", how="left")

    logger.info("Training LightGBM for AX, AY classes...")
    ax_ay_df = forecast_ax_ay(df_with_class)

    logger.info("Forecasting CX class with Historic Mean...")
    cx_df = forecast_cx(df_with_class)

    logger.info("Forecasting BZ, CZ classes with SeasonalNaive...")
    bz_cz_df = forecast_bz_cz(df_with_class)

    forecast_df = combine_and_save(ax_ay_df, cx_df, bz_cz_df)

    eval_result = evaluate_models(
        y_true=forecast_df["demand"],
        predictions={"Combined Forecast": forecast_df["forecast"]},
    )
    logger.info(f"Forecast evaluation:\n{eval_result.to_string()}")

    logger.info("=" * 50)
    logger.info("STEP 4 — Inventory policy simulation")
    logger.info("=" * 50)

    policy_df        = compute_policy_params(forecast_df, inventory_df, sku_class)
    old_policy       = compute_old_policy_metric(inventory_df, policy_df)
    sim_results      = simulate_new_policy(inventory_df, policy_df)
    new_policy       = compute_new_policy_metric(sim_results, policy_df, inventory_df)

    save_simulation_results(policy_df, old_policy, new_policy)

    logger.info("Policy comparison (avg by class):")
    compare = old_policy.groupby("class")[["fill_rate", 'avg_inventory', "total_cost"]].mean().add_prefix("old_").join(
        new_policy.groupby("class")[["fill_rate", 'avg_inventory', "total_cost"]].mean().add_prefix("new_")
    )
    logger.info(f"\n{compare.to_string()}")

    logger.info("=" * 50)
    logger.info("Pipeline complete. All outputs saved to data/processed/")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()