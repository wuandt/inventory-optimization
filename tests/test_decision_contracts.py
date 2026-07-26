"""Focused tests for forecasting windows and inventory-policy decisions.

All fixtures are constructed in memory.  These tests intentionally do not read
pipeline artifacts, so they can identify contract regressions before a full
pipeline refresh.
"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backend"))

from models.forecasting import (
    _nonnegative_forecast,
    combine_and_save,
    forecast_baseline_class,
    validation_forecast_baseline_class,
)
from models.simulation import (
    _simulate_sku,
    cost_assumption_sensitivity,
    estimate_historical_policy,
    estimate_inferred_historical_policy_sensitivity,
    optimize_policy_grid,
)


class TestDecisionContracts(unittest.TestCase):
    @staticmethod
    def _baseline_forecast_fixture() -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", "2025-10-03", freq="D")
        return pd.DataFrame(
            {
                "sku_id": "SKU1",
                "date": dates,
                "demand": 5.0 + (np.arange(len(dates)) % 7),
                "class": "CX",
            }
        )

    def test_selection_calibration_and_final_forecast_dates_are_disjoint(self):
        frame = self._baseline_forecast_fixture()

        selection, calibration, selected = validation_forecast_baseline_class(
            frame, ["CX"]
        )
        final = forecast_baseline_class(frame, ["CX"], selected=selected)

        selection_dates = set(selection["date"])
        calibration_dates = set(calibration["date"])
        final_dates = set(final["date"])
        self.assertTrue(selection_dates.isdisjoint(calibration_dates))
        self.assertTrue(selection_dates.isdisjoint(final_dates))
        self.assertTrue(calibration_dates.isdisjoint(final_dates))
        self.assertEqual(selection["date"].min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(selection["date"].max(), pd.Timestamp("2025-05-30"))
        self.assertEqual(calibration["date"].min(), pd.Timestamp("2025-05-31"))
        self.assertEqual(calibration["date"].max(), pd.Timestamp("2025-09-30"))
        self.assertEqual(final["date"].min(), pd.Timestamp("2025-10-01"))
        self.assertLess(selection["date"].max(), calibration["date"].min())
        self.assertLess(calibration["date"].max(), final["date"].min())

        for forecast in [selection, calibration, final]:
            self.assertTrue((forecast["forecast"] >= 0).all())

    def test_forecast_postprocessing_clips_and_output_guard_rejects_negatives(self):
        np.testing.assert_array_equal(
            _nonnegative_forecast(np.array([-3.5, 0.0, 2.5])),
            np.array([0.0, 0.0, 2.5]),
        )
        invalid = pd.DataFrame(
            {
                "sku_id": ["SKU1"],
                "date": [pd.Timestamp("2025-10-01")],
                "demand": [1.0],
                "forecast": [-0.1],
                "model": ["fixture"],
            }
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            combine_and_save(invalid)

    def test_recorded_rop_is_primary_and_receipt_inference_is_sensitivity(self):
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 4,
                "date": pd.date_range("2025-09-27", periods=4, freq="D"),
                "reorder_point": [40.0] * 4,
                "order_quantity": [20.0] * 4,
                "lead_time": [1] * 4,
                "unit_cost": [5.0] * 4,
                "unit_price": [9.0] * 4,
                "inventory_level": [15.0, 12.0, 7.0, 18.0],
                "order_received": [0.0, 0.0, 0.0, 12.0],
            }
        )
        sku_class = pd.DataFrame({"sku_id": ["SKU1"], "class": ["AX"]})
        target_skus = pd.Series(["SKU1"])

        recorded = estimate_historical_policy(
            inventory, sku_class, target_skus
        )
        inferred = estimate_inferred_historical_policy_sensitivity(
            inventory, sku_class, target_skus
        )

        self.assertEqual(recorded.loc[0, "rop"], 40.0)
        self.assertEqual(recorded.loc[0, "order_quantity"], 20.0)
        self.assertEqual(inferred.loc[0, "rop"], 7.0)
        self.assertEqual(inferred.loc[0, "order_quantity"], 12.0)
        self.assertNotEqual(recorded.loc[0, "rop"], inferred.loc[0, "rop"])

    def test_dynamic_rop_uses_t_plus_one_forecast_and_terminal_places_no_order(self):
        daily = pd.DataFrame(
            {
                "date": pd.date_range("2025-10-01", periods=3, freq="D"),
                "demand": [0.0, 0.0, 0.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 3,
                "date": pd.date_range("2025-10-01", periods=3, freq="D"),
                # The first value is deliberately large: the day-one decision
                # must ignore it and use the forecast targeting day two.
                "forecast": [100.0, 4.0, 7.0],
            }
        )
        policy = pd.Series(
            {
                "sku_id": "SKU1",
                "rop": 0.0,
                "order_quantity": 10.0,
                "lead_time": 2,
                "safety_stock": 3.0,
            }
        )

        result = _simulate_sku(
            daily,
            policy,
            initial_inventory=0.0,
            forecast_schedule=schedule,
        )

        self.assertEqual(result.loc[0, "rop_used"], 4.0 * 2 + 3.0)
        self.assertEqual(result.loc[1, "rop_used"], 7.0 * 2 + 3.0)
        self.assertEqual(
            result.loc[0, "forecast_target_date"], pd.Timestamp("2025-10-02")
        )
        self.assertEqual(
            result.loc[1, "forecast_target_date"], pd.Timestamp("2025-10-03")
        )
        self.assertEqual(result["order_placed"].tolist(), [1, 1, 0])
        self.assertEqual(
            result["order_decision_eligible"].tolist(), [True, True, False]
        )
        self.assertTrue(pd.isna(result.loc[2, "rop_used"]))
        self.assertTrue(pd.isna(result.loc[2, "forecast_target_date"]))

    def test_policy_grid_enforces_moq_and_order_multiple(self):
        calibration = pd.DataFrame(
            {
                "sku_id": ["SKU1"] * 5,
                "date": pd.date_range("2025-05-31", periods=5, freq="D"),
                "demand": [5.0] * 5,
                "forecast": [5.0] * 5,
            }
        )
        inventory = pd.DataFrame(
            {
                "sku_id": ["SKU1"],
                "date": [pd.Timestamp("2025-05-30")],
                "lead_time": [1],
                "unit_cost": [5.0],
                "unit_price": [9.0],
                "order_quantity": [5.0],
                "inventory_level": [0.0],
            }
        )
        sku_class = pd.DataFrame({"sku_id": ["SKU1"], "class": ["AX"]})

        selected, audit = optimize_policy_grid(
            calibration,
            inventory,
            sku_class,
            q_multipliers=(0.5, 2.0),
            safety_stock_quantiles=(0.5,),
            service_floor=0.0,
            minimum_order_quantity=7.0,
            order_multiple=4.0,
        )

        self.assertEqual(len(audit), 4)
        self.assertEqual(set(audit["order_quantity"]), {8.0, 12.0})
        self.assertTrue((audit["order_quantity"] >= 7.0).all())
        self.assertTrue(np.allclose(audit["order_quantity"] % 4.0, 0.0))
        self.assertIn(selected.loc[0, "order_quantity"], {8.0, 12.0})
        self.assertEqual(selected.loc[0, "selection_status"], "minimum_cost_feasible")
        feasible = audit[audit["service_floor_met"]]
        self.assertAlmostEqual(
            selected.loc[0, "calibration_total_cost"],
            feasible["total_cost"].min(),
        )

    def test_cost_sensitivity_reprices_same_paths_across_full_grid(self):
        dates = pd.date_range("2025-10-01", periods=2, freq="D")
        old_results = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU1"],
                "date": dates,
                "demand": [10.0, 10.0],
                "sales_quantity": [8.0, 8.0],
                "lost": [2.0, 2.0],
                "inventory": [5.0, 4.0],
                "order_placed": [1, 0],
                "received": [0.0, 0.0],
            }
        )
        new_results = pd.DataFrame(
            {
                "sku_id": ["SKU1", "SKU1"],
                "date": dates,
                "demand": [10.0, 10.0],
                "sales_quantity": [10.0, 10.0],
                "lost": [0.0, 0.0],
                "inventory": [8.0, 7.0],
                "order_placed": [1, 1],
                "received": [0.0, 0.0],
            }
        )
        policy = pd.DataFrame(
            {
                "sku_id": ["SKU1"],
                "class": ["AX"],
                "unit_cost": [5.0],
                "unit_price": [10.0],
            }
        )

        sensitivity = cost_assumption_sensitivity(
            old_results,
            new_results,
            policy,
            policy,
            ordering_costs=(10.0, 20.0),
            holding_rates=(0.1, 0.2),
            stockout_cost_multipliers=(0.5, 1.5),
        )

        self.assertEqual(len(sensitivity), 8)
        self.assertEqual(
            len(
                sensitivity[
                    [
                        "ordering_cost_per_order",
                        "annual_holding_rate",
                        "stockout_margin_multiplier",
                    ]
                ].drop_duplicates()
            ),
            8,
        )
        self.assertTrue(
            np.allclose(
                sensitivity["old_total_cost"],
                sensitivity["old_stockout_cost"]
                + sensitivity["old_holding_cost"]
                + sensitivity["old_ordering_cost"],
            )
        )
        self.assertTrue(
            np.allclose(
                sensitivity["new_total_cost"],
                sensitivity["new_stockout_cost"]
                + sensitivity["new_holding_cost"]
                + sensitivity["new_ordering_cost"],
            )
        )

        low = sensitivity[
            (sensitivity["ordering_cost_per_order"] == 10.0)
            & (sensitivity["annual_holding_rate"] == 0.1)
            & (sensitivity["stockout_margin_multiplier"] == 0.5)
        ].iloc[0]
        high_shortage = sensitivity[
            (sensitivity["ordering_cost_per_order"] == 10.0)
            & (sensitivity["annual_holding_rate"] == 0.1)
            & (sensitivity["stockout_margin_multiplier"] == 1.5)
        ].iloc[0]
        high_ordering = sensitivity[
            (sensitivity["ordering_cost_per_order"] == 20.0)
            & (sensitivity["annual_holding_rate"] == 0.1)
            & (sensitivity["stockout_margin_multiplier"] == 0.5)
        ].iloc[0]
        high_holding = sensitivity[
            (sensitivity["ordering_cost_per_order"] == 10.0)
            & (sensitivity["annual_holding_rate"] == 0.2)
            & (sensitivity["stockout_margin_multiplier"] == 0.5)
        ].iloc[0]

        self.assertAlmostEqual(
            high_shortage["old_stockout_cost"], 3 * low["old_stockout_cost"]
        )
        self.assertAlmostEqual(
            high_ordering["old_ordering_cost"], 2 * low["old_ordering_cost"]
        )
        self.assertAlmostEqual(
            high_ordering["new_ordering_cost"], 2 * low["new_ordering_cost"]
        )
        self.assertAlmostEqual(
            high_holding["old_holding_cost"], 2 * low["old_holding_cost"]
        )
        self.assertAlmostEqual(
            high_holding["new_holding_cost"], 2 * low["new_holding_cost"]
        )
        self.assertAlmostEqual(low["fill_rate_delta_percentage_points"], 20.0)


if __name__ == "__main__":
    unittest.main()