import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd

# Project paths and column names
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH, encoding="utf-8") as config_file:
    _config = json.load(config_file)

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_RAW_DIR = os.path.join(_ROOT, "data", "raw")
_PRO_DIR = os.path.join(_ROOT, "data", "processed")
_META_DIR = os.path.join(_ROOT, "data", "metadata")
_DATE_COL = _config["data"]["date_col"]
_SKU_COL = _config["data"]["sku_col"]

# Minimum columns expected in each file
_REQUIRED_RAW = [
    "sku_id",
    "date",
    "demand",
    "sales_quantity",
    "inventory_level",
    "lead_time",
    "unit_cost",
    "unit_price",
    "order_received",
    "category",
    "safety_stock",
    "reorder_point",
    "order_quantity",
]
_REQUIRED_COLUMNS = {
    "inventory_processed": _REQUIRED_RAW,
    "sku_class": ["sku_id", "class"],
    "sku_metric": ["sku_id", "fill_rate", "lost_sales", "DOI", "CV"],
    "forecast": ["sku_id", "date", "demand", "forecast"],
    "forecast_metrics": [
        "level",
        "segment",
        "MAE",
        "RMSE",
        "WAPE",
        "Bias",
        "MASE",
        "RMSSE",
        "n_obs",
    ],
    "policy_sku": [
        "sku_id",
        "class",
        "safety_stock",
        "rop",
        "order_quantity",
    ],
    "old_policy_metric": [
        "sku_id",
        "class",
        "total_demand",
        "total_sales",
        "fill_rate",
        "cycle_service_level",
        "avg_inventory",
        "total_cost",
    ],
    "new_policy_metric": [
        "sku_id",
        "class",
        "total_demand",
        "total_sales",
        "fill_rate",
        "cycle_service_level",
        "avg_inventory",
        "total_cost",
    ],
    "policy_uncertainty": [
        "metric", "estimate", "ci_2_5", "ci_97_5", "n_bootstrap", "method"
    ],
    "optimization_scope": ["class", "intervention"],
}

# Extra analysis files shown on the policy page
DASHBOARD_LINEAGE_ARTIFACTS = {
    "policy_candidate_audit": "data/processed/policy_candidate_audit.csv",
    "policy_sensitivity": "data/processed/policy_sensitivity.csv",
    "policy_action": "data/processed/policy_action.csv",
    "full_policy_summary": "data/processed/full_policy_summary.csv",
    "scenario_uncertainty": "data/processed/scenario_uncertainty.csv",
    "historical_policy_sensitivity": (
        "data/processed/historical_policy_sensitivity.csv"
    ),
}

# Minimum fields used by the policy page
DASHBOARD_ARTIFACT_REQUIRED_COLUMNS = {
    "policy_candidate_audit": [
        "sku_id",
        "class",
        "q_multiplier",
        "safety_stock_quantile",
        "order_quantity",
        "safety_stock",
        "rop",
        "fill_rate",
        "total_cost",
        "service_floor_met",
    ],
    "policy_sensitivity": [
        "scenario_id",
        "ordering_cost_per_order",
        "annual_holding_rate",
        "stockout_margin_multiplier",
        "total_cost_savings",
    ],
    "policy_action": [
        "sku_id",
        "class",
        "recommended_action",
        "current_rop",
        "proposed_rop",
        "current_order_quantity",
        "proposed_order_quantity",
        "selection_status",
    ],
    "historical_policy_sensitivity": [
        "baseline_source",
        "mean_rop",
        "fill_rate",
        "total_cost",
    ],
    "full_policy_summary": [
        "policy_label",
        "sku_count",
        "total_demand",
        "total_sales",
        "total_lost",
        "fill_rate",
        "cycle_service_level",
        "sum_sku_avg_inventory",
        "stockout_cost",
        "holding_cost",
        "ordering_cost",
        "total_cost",
    ],
    "scenario_uncertainty": [
        "scenario_id",
        "lead_time_multiplier",
        "old_fill_rate",
        "new_fill_rate",
        "fill_rate_delta_percentage_points",
        "avg_inventory_change_pct",
        "old_total_cost",
        "new_total_cost",
        "total_cost_savings",
        "policy_contract_semantics",
        "evaluation_semantics",
    ],
}

MANIFEST_REQUIRED_ARTIFACTS = (
    "data/raw/inventory.csv",
    "src/backend/config/config.json",
    "data/processed/inventory_processed.csv",
    "data/metadata/inventory_data_contract.json",
    "data/processed/sku_class.csv",
    "data/processed/sku_metric.csv",
    "data/processed/optimization_scope.csv",
    "data/processed/validation_forecast.csv",
    "data/processed/forecast.csv",
    "data/processed/forecast_metrics.csv",
    "data/processed/policy_sku.csv",
    "data/processed/old_policy_metric.csv",
    "data/processed/new_policy_metric.csv",
    "data/processed/policy_uncertainty.csv",
    "data/processed/policy_candidate_audit.csv",
    "data/processed/policy_sensitivity.csv",
    "data/processed/policy_action.csv",
    "data/processed/historical_policy_sensitivity.csv",
    "data/processed/full_policy_summary.csv",
    "data/processed/scenario_uncertainty.csv",
    "params/best_params.json",
    "params/feature_importance.csv",
)


class DataLoader:
    def __init__(self, mode: str = "dashboard"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.mode = mode

        self.inventory = None
        self.sku_class = None
        self.sku_metric = None
        self.forecast = None
        self.forecast_metrics = None
        self.policy_sku = None
        self.old_policy = None
        self.new_policy = None
        self.policy_uncertainty = None
        self.policy_candidate_audit = None
        self.policy_sensitivity = None
        self.policy_action = None
        self.full_policy_summary = None
        self.scenario_uncertainty = None
        self.historical_policy_sensitivity = None
        self.optimization_scope = None
        self.artifact_manifest = None

    def load(self) -> "DataLoader":
        """Load either the pipeline input or the dashboard outputs."""
        if self.mode == "pipeline":
            return self._load_pipeline()
        return self._load_dashboard()

    def _load_pipeline(self) -> "DataLoader":
        """Load and validate the fixed synthetic input dataset."""
        raw_path = os.path.join(_RAW_DIR, "inventory.csv")
        try:
            raw_inventory = pd.read_csv(raw_path, parse_dates=[_DATE_COL])
            self.logger.info("Raw data loaded: %s", raw_inventory.shape)
        except Exception as error:
            self.logger.error("Error loading raw data: %s", error)
            raise

        missing = [
            column
            for column in _REQUIRED_RAW
            if column not in raw_inventory.columns
        ]
        if missing:
            raise ValueError(f"Missing columns in raw data: {missing}")

        self.inventory = self._preprocess(raw_inventory)
        self.save_processed(self.inventory)
        return self

    def _preprocess(self, inventory: pd.DataFrame) -> pd.DataFrame:
        """Validate the input data contract without silently repairing rows."""
        inventory = inventory.copy()
        inventory[_DATE_COL] = pd.to_datetime(inventory[_DATE_COL])
        inventory = inventory.sort_values(
            [_SKU_COL, _DATE_COL]
        ).reset_index(drop=True)

        self._check_sku_date_key(inventory)
        inventory["lead_time"] = self._validated_lead_time(inventory)

        if inventory.isna().any().any():
            raise ValueError(
                "Data contract violation: missing values require explicit treatment."
            )

        self._check_daily_calendar(inventory)
        self._check_nonnegative_values(inventory)
        self._check_inventory_balance(inventory)

        self.logger.info("Preprocessed data: %s", inventory.shape)
        return inventory

    @staticmethod
    def _check_sku_date_key(inventory: pd.DataFrame) -> None:
        """Each row must have one unique SKU-date key."""
        key_columns = [_SKU_COL, _DATE_COL]
        if inventory[key_columns].isna().any().any():
            raise ValueError("Data contract violation: missing SKU-date key.")
        if inventory.duplicated(key_columns).any():
            raise ValueError("Data contract violation: duplicate SKU-date rows.")

    @staticmethod
    def _validated_lead_time(inventory: pd.DataFrame) -> pd.Series:
        """Return lead time as positive whole days."""
        lead_time = pd.to_numeric(inventory["lead_time"], errors="coerce")
        invalid_lead_time = (
            ~np.isfinite(lead_time)
            | (lead_time <= 0)
            | ~np.isclose(lead_time, np.round(lead_time))
        )
        if invalid_lead_time.any():
            examples = (
                inventory.loc[
                    invalid_lead_time,
                    [_SKU_COL, _DATE_COL, "lead_time"],
                ]
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                "Data contract violation: lead_time must be finite, positive, "
                f"integer-valued days. Examples: {examples}"
            )
        return lead_time.astype("int64")

    @staticmethod
    def _check_daily_calendar(inventory: pd.DataFrame) -> None:
        """Each SKU must have exactly one row for every calendar day."""
        normalized_dates = inventory[_DATE_COL].dt.normalize()
        if not normalized_dates.eq(inventory[_DATE_COL]).all():
            raise ValueError(
                "Data contract violation: date must be a normalized calendar day."
            )

        previous_date = inventory.groupby(
            _SKU_COL, observed=True
        )[_DATE_COL].shift(1)
        day_gap = (inventory[_DATE_COL] - previous_date).dt.days
        discontinuous = previous_date.notna() & day_gap.ne(1)
        if discontinuous.any():
            examples = (
                inventory.loc[discontinuous, [_SKU_COL, _DATE_COL]]
                .assign(previous_date=previous_date[discontinuous].to_numpy())
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                "Data contract violation: each SKU calendar must contain one "
                f"continuous row per day. Examples: {examples}"
            )

    @staticmethod
    def _check_nonnegative_values(inventory: pd.DataFrame) -> None:
        """Operational quantities and costs cannot be negative."""
        nonnegative_columns = [
            "demand",
            "sales_quantity",
            "inventory_level",
            "order_received",
            "lead_time",
            "safety_stock",
            "reorder_point",
            "order_quantity",
            "unit_cost",
            "unit_price",
        ]
        if (inventory[nonnegative_columns] < 0).any().any():
            raise ValueError(
                "Data contract violation: negative operational quantity or cost."
            )
        if (inventory["sales_quantity"] > inventory["demand"]).any():
            raise ValueError("Data contract violation: sales_quantity exceeds demand.")

    @staticmethod
    def _check_inventory_balance(inventory: pd.DataFrame) -> None:
        """Check: ending stock = prior ending stock + receipts - sales."""
        prior_inventory = inventory.groupby(
            _SKU_COL, observed=True
        )["inventory_level"].shift(1)
        rows_with_prior_day = prior_inventory.notna()

        expected_inventory = (
            prior_inventory[rows_with_prior_day]
            + inventory.loc[rows_with_prior_day, "order_received"]
            - inventory.loc[rows_with_prior_day, "sales_quantity"]
        )
        actual_inventory = inventory.loc[
            rows_with_prior_day, "inventory_level"
        ]
        if not actual_inventory.eq(expected_inventory).all():
            raise ValueError(
                "Data contract violation: end-of-day inventory conservation fails."
            )

    def save_processed(
        self,
        inventory: pd.DataFrame,
        filename: str = "inventory_processed.csv",
    ) -> None:
        """Save a processed CSV and its data contract when applicable."""
        os.makedirs(_PRO_DIR, exist_ok=True)
        path = os.path.join(_PRO_DIR, filename)
        inventory.to_csv(path, index=False)

        if filename == "inventory_processed.csv":
            os.makedirs(_META_DIR, exist_ok=True)
            contract = self._build_data_contract(inventory, filename)
            contract_path = os.path.join(
                _META_DIR, "inventory_data_contract.json"
            )
            with open(contract_path, "w", encoding="utf-8") as contract_file:
                json.dump(contract, contract_file, indent=2)

        self.logger.info("Saved -> %s", path)

    @staticmethod
    def _build_data_contract(
        inventory: pd.DataFrame,
        filename: str,
    ) -> dict:
        """Describe the processed inventory file in plain metadata."""
        schema = {
            column: str(dtype)
            for column, dtype in inventory.dtypes.items()
        }
        return {
            "artifact": f"data/processed/{filename}",
            "grain": "one row per sku_id per calendar date",
            "inventory_level_convention": "end_of_day",
            "target": _config["data"]["target_col"],
            "primary_key": [_SKU_COL, _DATE_COL],
            "schema": schema,
            "date_range": {
                "start": str(inventory[_DATE_COL].min().date()),
                "end": str(inventory[_DATE_COL].max().date()),
            },
            "final_holdout": {
                "start": _config["data"]["final_test_start"],
                "purpose": (
                    "locked retrospective evaluation; excluded from "
                    "selection and calibration"
                ),
            },
            "row_count": int(len(inventory)),
            "sku_count": int(inventory[_SKU_COL].nunique()),
        }

    def _load_dashboard(self) -> "DataLoader":
        """Load all files used by the four dashboard pages."""
        try:
            self.inventory = self._read_processed_csv(
                "inventory_processed.csv",
                parse_dates=[_DATE_COL],
            )
            self.sku_class = self._read_processed_csv("sku_class.csv")
            self.sku_metric = self._read_processed_csv("sku_metric.csv")
            self.forecast = self._read_processed_csv(
                "forecast.csv",
                parse_dates=[_DATE_COL],
            )
            self.forecast_metrics = self._read_processed_csv(
                "forecast_metrics.csv"
            )
            self.policy_sku = self._read_processed_csv("policy_sku.csv")
            self.old_policy = self._read_processed_csv(
                "old_policy_metric.csv"
            )
            self.new_policy = self._read_processed_csv(
                "new_policy_metric.csv"
            )
            self.policy_uncertainty = self._read_processed_csv(
                "policy_uncertainty.csv"
            )
            self.optimization_scope = self._read_processed_csv(
                "optimization_scope.csv"
            )

            self.policy_candidate_audit = self._read_dashboard_artifact(
                DASHBOARD_LINEAGE_ARTIFACTS["policy_candidate_audit"]
            )
            self.policy_sensitivity = self._read_dashboard_artifact(
                DASHBOARD_LINEAGE_ARTIFACTS["policy_sensitivity"]
            )
            self.policy_action = self._read_dashboard_artifact(
                DASHBOARD_LINEAGE_ARTIFACTS["policy_action"]
            )
            self.full_policy_summary = self._read_dashboard_artifact(
                DASHBOARD_LINEAGE_ARTIFACTS["full_policy_summary"]
            )
            self.scenario_uncertainty = self._read_dashboard_artifact(
                DASHBOARD_LINEAGE_ARTIFACTS["scenario_uncertainty"]
            )
            self.historical_policy_sensitivity = (
                self._read_dashboard_artifact(
                    DASHBOARD_LINEAGE_ARTIFACTS[
                        "historical_policy_sensitivity"
                    ]
                )
            )

            self.artifact_manifest = self._read_manifest()
            self.logger.info("Dashboard data loaded successfully.")
        except Exception as error:
            self.logger.error("Error loading dashboard data: %s", error)
            raise

        if not self.validate():
            raise ValueError("Data validation failed — check logs for details.")

        return self

    @staticmethod
    def _read_processed_csv(
        filename: str,
        parse_dates: list = None,
    ) -> pd.DataFrame:
        """Read one canonical file from data/processed."""
        path = os.path.join(_PRO_DIR, filename)
        return pd.read_csv(path, parse_dates=parse_dates)

    @staticmethod
    def _read_manifest():
        """Read the artifact manifest when it exists."""
        manifest_path = os.path.join(_META_DIR, "artifact_manifest.json")
        if not os.path.isfile(manifest_path):
            return None

        with open(manifest_path, encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def _read_dashboard_artifact(self, relative_path: str) -> pd.DataFrame:
        """Read a lineage artifact and defer consolidated failures to validate()."""
        path = os.path.join(_ROOT, *relative_path.split("/"))
        if not os.path.isfile(path):
            self.logger.error(
                "Required dashboard artifact not found: %s", relative_path
            )
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as error:
            self.logger.error(
                "Required dashboard artifact could not be loaded (%s): %s",
                relative_path,
                error,
            )
            return pd.DataFrame()

    def validate(self) -> bool:
        """Check schemas first, then check dashboard file lineage."""
        core_files_are_valid = self._validate_core_files()
        if self.mode == "pipeline":
            return core_files_are_valid

        lineage_files_are_valid = self._validate_lineage_files()
        manifest_is_valid = self._validate_manifest()
        return (
            core_files_are_valid
            and lineage_files_are_valid
            and manifest_is_valid
        )

    def _validate_core_files(self) -> bool:
        """Check that every main dataset is loaded with required columns."""
        loaded_files = {
            "inventory_processed": self.inventory,
            "sku_class": self.sku_class,
            "sku_metric": self.sku_metric,
            "forecast": self.forecast,
            "forecast_metrics": self.forecast_metrics,
            "policy_sku": self.policy_sku,
            "old_policy_metric": self.old_policy,
            "new_policy_metric": self.new_policy,
            "policy_uncertainty": self.policy_uncertainty,
            "optimization_scope": self.optimization_scope,
        }

        valid = True
        for file_name, data in loaded_files.items():
            if data is None:
                self.logger.error("%s is not loaded.", file_name)
                valid = False
                continue

            required_columns = _REQUIRED_COLUMNS[file_name]
            missing = [
                column
                for column in required_columns
                if column not in data.columns
            ]
            if missing:
                self.logger.error(
                    "Missing columns in %s: %s",
                    file_name,
                    missing,
                )
                valid = False
        return valid

    def _validate_lineage_files(self) -> bool:
        """Check the extra policy evidence used by the dashboard."""
        valid = True
        for file_name in DASHBOARD_LINEAGE_ARTIFACTS:
            data = getattr(self, file_name)
            if data is None or data.empty:
                self.logger.error(
                    "Required dashboard artifact %s is not loaded or is empty.",
                    file_name,
                )
                valid = False
                continue

            required_columns = DASHBOARD_ARTIFACT_REQUIRED_COLUMNS[file_name]
            missing = [
                column
                for column in required_columns
                if column not in data.columns
            ]
            if missing:
                self.logger.error(
                    "Required dashboard artifact %s is missing columns: %s",
                    file_name,
                    missing,
                )
                valid = False
        return valid

    def _validate_manifest(self) -> bool:
        """Verify that dashboard files match the recorded pipeline run."""
        if not self.artifact_manifest:
            self.logger.error("artifact_manifest.json is missing.")
            return False

        artifacts = self.artifact_manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            self.logger.error(
                "artifact_manifest.json must contain an artifacts object."
            )
            return False

        valid = self._validate_manifest_membership(artifacts)
        for relative_path in MANIFEST_REQUIRED_ARTIFACTS:
            if relative_path not in artifacts:
                continue

            file_is_valid = self._validate_manifest_file(
                relative_path,
                artifacts[relative_path],
            )
            if not file_is_valid:
                valid = False
        return valid

    def _validate_manifest_membership(self, artifacts: dict) -> bool:
        """The manifest must contain exactly the documented files."""
        expected_files = set(MANIFEST_REQUIRED_ARTIFACTS)
        actual_files = set(artifacts)
        missing_files = sorted(expected_files - actual_files)
        unexpected_files = sorted(actual_files - expected_files)

        valid = True
        if missing_files:
            self.logger.error(
                "Manifest is missing required artifacts: %s",
                ", ".join(missing_files),
            )
            valid = False
        if unexpected_files:
            self.logger.error(
                "Manifest contains unexpected artifacts: %s",
                ", ".join(unexpected_files),
            )
            valid = False
        return valid

    def _validate_manifest_file(
        self,
        relative_path: str,
        metadata: dict,
    ) -> bool:
        """Validate one file's existence, size and SHA-256 checksum."""
        if not isinstance(metadata, dict):
            self.logger.error(
                "Invalid manifest metadata for artifact: %s",
                relative_path,
            )
            return False

        absolute_path = os.path.join(_ROOT, *relative_path.split("/"))
        if not os.path.isfile(absolute_path):
            self.logger.error(
                "Manifest artifact is missing: %s",
                relative_path,
            )
            return False

        valid = True
        expected_bytes = metadata.get("bytes")
        actual_bytes = os.path.getsize(absolute_path)
        if not isinstance(expected_bytes, int) or expected_bytes != actual_bytes:
            self.logger.error(
                "Artifact size mismatch: %s. Run the full backend pipeline.",
                relative_path,
            )
            valid = False

        expected_digest = metadata.get("sha256")
        actual_digest = self._calculate_sha256(absolute_path)
        if (
            not isinstance(expected_digest, str)
            or expected_digest != actual_digest
        ):
            self.logger.error(
                "Stale artifact detected: %s. Run the full backend pipeline.",
                relative_path,
            )
            valid = False
        return valid

    @staticmethod
    def _calculate_sha256(path: str) -> str:
        """Calculate a file checksum in small chunks."""
        digest = hashlib.sha256()
        with open(path, "rb") as source_file:
            while True:
                chunk = source_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def get_class_summary(self) -> pd.DataFrame:
        sku_data = self.sku_metric.merge(
            self.sku_class,
            on=_SKU_COL,
            how="left",
        )
        summary = (
            sku_data.groupby("class", observed=True)
            .agg(
                sku_count=(_SKU_COL, "count"),
                total_sales=("total_sales", "sum"),
                total_demand=("total_demand", "sum"),
                avg_lost_sales=("lost_sales", "mean"),
                avg_doi=("DOI", "mean"),
            )
            .reset_index()
        )
        summary["avg_fill_rate"] = summary["total_sales"] / summary["total_demand"]
        return summary[
            ["class", "sku_count", "avg_fill_rate", "avg_lost_sales", "avg_doi"]
        ].round(3)

    def get_forecast_by_sku(self, sku_id: str) -> pd.DataFrame:
        return (
            self.forecast[self.forecast[_SKU_COL] == sku_id]
            .sort_values(_DATE_COL)
            .reset_index(drop=True)
        )

    def get_policy_comparison(self) -> pd.DataFrame:
        """Compare old and proposed policy metrics by ABC-XYZ class."""
        old_summary = self._summarize_policy_by_class(
            self.old_policy,
            prefix="old_",
        )
        new_summary = self._summarize_policy_by_class(
            self.new_policy,
            prefix="new_",
        )
        return old_summary.join(new_summary).reset_index()

    @staticmethod
    def _summarize_policy_by_class(
        policy_data: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """Add SKU policy outcomes within each class."""
        summary = policy_data.groupby("class", observed=True).agg(
            total_sales=("total_sales", "sum"),
            total_demand=("total_demand", "sum"),
            avg_inventory=("avg_inventory", "sum"),
            holding_cost=("holding_cost", "sum"),
            ordering_cost=("ordering_cost", "sum"),
            stockout_cost=("stockout_cost", "sum"),
            total_cost=("total_cost", "sum"),
        )
        summary["fill_rate"] = (
            summary["total_sales"] / summary["total_demand"]
        )
        output_columns = [
            "fill_rate",
            "avg_inventory",
            "holding_cost",
            "ordering_cost",
            "stockout_cost",
            "total_cost",
        ]
        return summary[output_columns].add_prefix(prefix)

    def get_inventory_trend(self) -> pd.DataFrame:
        return (
            self.inventory.groupby(_DATE_COL)["inventory_level"]
            .sum()
            .reset_index()
            .rename(columns={"inventory_level": "total_inventory"})
        )

    def get_policy_sku(self, sku_id: str = None) -> pd.DataFrame:
        if sku_id:
            selected_policy = self.policy_sku[
                self.policy_sku[_SKU_COL] == sku_id
            ]
            return selected_policy.reset_index(drop=True)
        return self.policy_sku

    def get_skus_by_class(self, cls: str) -> list:
        selected_class = self.sku_class["class"] == cls
        return self.sku_class.loc[selected_class, _SKU_COL].tolist()