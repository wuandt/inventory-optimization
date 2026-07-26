"""Regression tests for the leakage-safe forecasting and simulation contracts."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backend"))

from features.feature_engineering import create_features
from data.loader import DataLoader, MANIFEST_REQUIRED_ARTIFACTS
import main as backend_main
from models.forecasting import _date_folds
from models.simulation import (
    _simulate_sku,
    bootstrap_policy_comparison,
    compute_policy_params,
    simulate_policy,
)
from models.classification import (
    classify_abc,
    classify_xyz,
    merge_abc_xyz,
    select_optimization_scope,
)


class TestCoreContracts(unittest.TestCase):
    def test_loader_rejects_duplicate_sku_day_instead_of_dropping_it(self):
        fixture = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU1"], "date": ["2025-01-01", "2025-01-01"],
                "demand": [1, 1], "sales_quantity": [1, 1], "inventory_level": [10, 10],
                "order_received": [0, 0], "category": ["Cat", "Cat"], "lead_time": [1, 1],
                "safety_stock": [1, 1], "reorder_point": [2, 2], "order_quantity": [5, 5],
                "unit_cost": [2.0, 2.0], "unit_price": [4.0, 4.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate SKU-date"):
            DataLoader(mode="pipeline")._preprocess(fixture)

    def test_notebook_feature_function_matches_backend_schema_and_values(self):
        """The feature implementation matches the backend contract."""
        notebook = json.loads((ROOT / "notebooks" / "04_forecasting.ipynb").read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def create_features(frame):" in "".join(cell["source"])
        )
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_features")
        notebook_namespace = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_04_features", "exec"), notebook_namespace)

        dates = pd.date_range("2024-01-01", periods=125, freq="D")
        fixture = pd.concat(
            [
                pd.DataFrame({"sku_id": sku, "date": dates, "demand": (np.arange(len(dates)) * multiplier) % 23})
                for sku, multiplier in [("SKU1", 1), ("SKU2", 3)]
            ],
            ignore_index=True,
        )
        expected = create_features(fixture)
        actual = notebook_namespace["create_features"](fixture)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_features_are_invariant_to_future_target_changes(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        base = pd.DataFrame({"sku_id": "SKU1", "date": dates, "demand": np.arange(120) % 17})
        changed = base.copy()
        changed.loc[changed["date"] > pd.Timestamp("2024-04-01"), "demand"] += 10_000

        base_features = create_features(base)
        changed_features = create_features(changed)
        cutoff = pd.Timestamp("2024-04-01")
        feature_cols = [column for column in base_features if column not in ["date", "demand"]]
        pd.testing.assert_frame_equal(
            base_features.loc[base_features["date"] <= cutoff, feature_cols].reset_index(drop=True),
            changed_features.loc[changed_features["date"] <= cutoff, feature_cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_date_folds_never_train_after_validation(self):
        dates = pd.date_range("2024-01-01", "2025-09-30", freq="D")
        frame = pd.DataFrame({"sku_id": "SKU1", "date": dates, "demand": 1.0})
        for start, end in _date_folds(frame):
            self.assertLess(start, end)
            self.assertGreaterEqual(start, pd.Timestamp("2025-01-01"))
            self.assertLess(end, pd.Timestamp("2025-10-01"))

    def test_segmentation_and_scope_are_invariant_to_calibration_and_final_demand(self):
        inventory = pd.read_csv(
            ROOT / "data" / "processed" / "inventory_processed.csv",
            parse_dates=["date"],
        )
        changed = inventory.copy()
        changed.loc[
            changed["date"] >= pd.Timestamp("2025-05-31"), "demand"
        ] += 10_000
        class_original = merge_abc_xyz(
            classify_abc(inventory), classify_xyz(inventory)
        ).sort_values("sku_id").reset_index(drop=True)
        class_changed = merge_abc_xyz(
            classify_abc(changed), classify_xyz(changed)
        ).sort_values("sku_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(class_original, class_changed)
        scope_original, _ = select_optimization_scope(inventory, class_original)
        scope_changed, _ = select_optimization_scope(changed, class_changed)
        pd.testing.assert_frame_equal(
            scope_original.sort_values("class").reset_index(drop=True),
            scope_changed.sort_values("class").reset_index(drop=True),
        )

    def test_simulator_conserves_inventory(self):
        daily = pd.DataFrame(
            {"date": pd.date_range("2025-10-01", periods=4, freq="D"), "demand": [5, 7, 3, 4]}
        )
        policy = pd.Series({"sku_id": "SKU1", "rop": 8, "order_quantity": 10, "lead_time": 1})
        result = _simulate_sku(daily, policy, initial_inventory=20)
        previous = 20.0
        for _, row in result.iterrows():
            self.assertAlmostEqual(row["inventory"], previous + row["received"] - row["sales_quantity"])
            self.assertGreaterEqual(row["inventory"], 0)
            previous = row["inventory"]

    def test_order_arrives_at_start_of_exact_lead_time_period(self):
        for lead_time in [1, 3, 5]:
            daily = pd.DataFrame(
                {
                    "date": pd.date_range("2025-10-01", periods=lead_time + 2),
                    "demand": 0.0,
                }
            )
            policy = pd.Series(
                {
                    "sku_id": "SKU1",
                    "rop": 0.0,
                    "order_quantity": 10.0,
                    "lead_time": lead_time,
                }
            )
            result = _simulate_sku(daily, policy, initial_inventory=0.0)
            first_receipt = result.index[result["received"] > 0].min()
            self.assertEqual(first_receipt, lead_time)

    def test_policy_calibration_rejects_final_holdout_rows(self):
        validation = pd.DataFrame(
            {"sku_id": ["SKU1"], "date": [pd.Timestamp("2025-10-01")], "demand": [10.0], "forecast": [9.0]}
        )
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1"], "date": [pd.Timestamp("2025-09-30")], "lead_time": [2],
                "unit_cost": [5.0], "unit_price": [10.0], "order_quantity": [20.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "configured"):
            compute_policy_params(validation, inventory, pd.DataFrame({"sku_id": ["SKU1"], "class": ["AX"]}))

    def test_dynamic_scope_service_level_has_intervention_fallback(self):
        validation = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU1"], "date": pd.to_datetime(["2025-05-31", "2025-06-01"]),
                "demand": [10.0, 12.0], "forecast": [9.0, 13.0],
            }
        )
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU1"], "date": pd.to_datetime(["2025-01-01", "2025-09-30"]),
                "lead_time": [2, 2], "unit_cost": [5.0, 5.0], "unit_price": [10.0, 10.0],
                "order_quantity": [20.0, 20.0],
            }
        )
        policy = compute_policy_params(
            validation,
            inventory,
            pd.DataFrame({"sku_id": ["SKU1"], "class": ["DX"]}),
            pd.DataFrame({"class": ["DX"], "intervention": ["protect_strategic_value"]}),
        )
        self.assertEqual(policy.loc[0, "target_service_level"], 97)

    def test_policy_uses_empirical_lead_time_error_and_ignores_final_demand(self):
        validation = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 4,
                "date": pd.date_range("2025-05-31", periods=4),
                "demand": [11.0, 12.0, 13.0, 14.0],
                "forecast": [10.0, 10.0, 10.0, 10.0],
            }
        )
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 3,
                "date": pd.to_datetime(["2025-01-01", "2025-09-30", "2025-10-01"]),
                "lead_time": [2, 2, 2],
                "unit_cost": [5.0, 5.0, 5.0],
                "unit_price": [10.0, 10.0, 10.0],
                "order_quantity": [20.0, 20.0, 20.0],
                "demand": [1.0, 1.0, 1.0],
            }
        )
        class_df = pd.DataFrame({"sku_id": ["SKU1"], "class": ["AY"]})
        scope_df = pd.DataFrame(
            {"class": ["AY"], "intervention": ["protect_strategic_value"]}
        )
        original = compute_policy_params(
            validation, inventory, class_df, scope_df
        )
        changed = inventory.copy()
        changed.loc[changed["date"] >= pd.Timestamp("2025-10-01"), "demand"] = 100_000
        changed_policy = compute_policy_params(
            validation, changed, class_df, scope_df
        )
        pd.testing.assert_frame_equal(original, changed_policy)
        expected_errors = np.array([3.0, 5.0, 7.0])
        self.assertAlmostEqual(
            original.loc[0, "safety_stock"],
            np.quantile(expected_errors, 0.97),
        )

    def test_policy_bootstrap_is_paired_deterministic_and_uses_portfolio_rates(self):
        old = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU2"],
                "total_demand": [100.0, 10.0],
                "total_sales": [90.0, 10.0],
                "avg_inventory": [80.0, 20.0],
                "total_cost": [900.0, 100.0],
            }
        )
        new = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU2"],
                "total_demand": [100.0, 10.0],
                "total_sales": [95.0, 10.0],
                "avg_inventory": [72.0, 18.0],
                "total_cost": [720.0, 80.0],
            }
        )
        first = bootstrap_policy_comparison(old, new, n_bootstrap=100, seed=7)
        second = bootstrap_policy_comparison(old, new, n_bootstrap=100, seed=7)
        pd.testing.assert_frame_equal(first, second)
        estimates = first.set_index("metric")["estimate"]
        self.assertAlmostEqual(
            estimates["fill_rate_delta_percentage_points"],
            100 * (105 / 110 - 100 / 110),
        )
        self.assertAlmostEqual(estimates["avg_inventory_change_pct"], -10.0)
        self.assertAlmostEqual(estimates["total_cost_change_pct"], -20.0)

    def test_both_policies_receive_identical_final_demand_and_initial_inventory(self):
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 4,
                "date": pd.to_datetime(["2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03"]),
                "demand": [0.0, 4.0, 8.0, 6.0], "inventory_level": [20.0, 0.0, 0.0, 0.0],
            }
        )
        base = {"sku_id": ["SKU1"], "class": ["AX"], "lead_time": [1], "unit_cost": [5.0], "unit_price": [10.0], "order_quantity": [10.0]}
        old_policy = pd.DataFrame({**base, "rop": [8.0]})
        new_policy = pd.DataFrame({**base, "rop": [4.0]})
        old = simulate_policy(inventory, old_policy)
        new = simulate_policy(inventory, new_policy)
        self.assertEqual(
            set(old[["sku_id", "date"]].itertuples(index=False, name=None)),
            set(new[["sku_id", "date"]].itertuples(index=False, name=None)),
        )
        self.assertEqual(old.loc[0, "initial_inventory"], new.loc[0, "initial_inventory"])

    def test_analysis_notebooks_are_read_only_consumers_of_canonical_artifacts(self):
        """Notebooks explain one backend run instead of creating rival outputs."""
        expected_artifacts = {
            "01_preprocessing.ipynb": [
                "inventory.csv",
                "inventory_processed.csv",
                "inventory_data_contract.json",
                "artifact_manifest.json",
            ],
            "03_ABC_XYZ.ipynb": [
                "sku_class.csv",
                "sku_metric.csv",
                "optimization_scope.csv",
                "config.json",
            ],
            "04_forecasting.ipynb": [
                "validation_forecast.csv",
                "forecast.csv",
                "forecast_metrics.csv",
                "best_params.json",
            ],
            "05_simulation.ipynb": [
                "policy_sku.csv",
                "policy_candidate_audit.csv",
                "policy_action.csv",
                "policy_sensitivity.csv",
                "scenario_uncertainty.csv",
            ],
        }
        for notebook_name, expected_names in expected_artifacts.items():
            notebook = json.loads(
                (ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8")
            )
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertNotIn(
                "to_csv(",
                code,
                f"{notebook_name} must not overwrite canonical pipeline outputs.",
            )
            self.assertNotIn("REFERENCE_DIR", code)
            for artifact_name in expected_names:
                self.assertIn(artifact_name, code)

    def test_manifest_generation_rejects_missing_required_artifacts(self):
        with mock.patch.object(backend_main.os.path, "isfile", return_value=False):
            with self.assertRaisesRegex(
                FileNotFoundError, "missing required artifacts"
            ):
                backend_main.save_artifact_manifest(
                    xyz_model="HistoricMean",
                    strategic_model="LightGBM",
                    understock_model="HistoricMean",
                    overstock_model="HistoricMean",
                    optimization_scope=pd.DataFrame(),
                    forecast_metrics=pd.DataFrame(),
                    old_metric=pd.DataFrame(),
                    new_metric=pd.DataFrame(),
                    policy_uncertainty=pd.DataFrame(),
                )

    def test_manifest_hashes_and_dashboard_portfolio_coverage(self):
        manifest_path = ROOT / "data" / "metadata" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest["artifacts"]),
            set(MANIFEST_REQUIRED_ARTIFACTS),
            "Manifest artifact membership must match the loader contract exactly.",
        )
        for relative, metadata in manifest["artifacts"].items():
            path = ROOT / Path(relative)
            self.assertTrue(path.exists(), relative)
            self.assertEqual(path.stat().st_size, metadata["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
        for relative, expected_digest in manifest["provenance"][
            "source_sha256"
        ].items():
            source_path = ROOT / Path(relative)
            self.assertTrue(source_path.exists(), relative)
            self.assertEqual(
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                expected_digest,
                f"Source lineage is stale for {relative}; rerun the pipeline.",
            )

        attributes = subprocess.run(
            [
                "git",
                "check-attr",
                "text",
                "--",
                *MANIFEST_REQUIRED_ARTIFACTS,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in attributes.stdout.splitlines():
            relative, attribute, value = line.rsplit(": ", 2)
            self.assertEqual(attribute, "text")
            self.assertEqual(
                value,
                "unset",
                f"{relative} must use -text so Git cannot change its manifest hash.",
            )

        loader = DataLoader(mode="dashboard").load()
        unit_cost = loader.inventory.groupby("sku_id")["unit_cost"].median()
        self.assertEqual(loader.sku_metric["sku_id"].nunique(), unit_cost.index.nunique())
        self.assertTrue(loader.sku_metric["sku_id"].isin(unit_cost.index).all())


if __name__ == "__main__":
    unittest.main()