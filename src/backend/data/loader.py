import json
import logging
import os

import pandas as pd

# ── Load config ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.json")
with open(_CONFIG_PATH) as f:
    _config = json.load(f)

_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RAW_DIR  = os.path.join(_ROOT, "data", "raw")
_PRO_DIR  = os.path.join(_ROOT, "data", "processed")
_DATE_COL = _config["data"]["date_col"]
_SKU_COL  = _config["data"]["sku_col"]

# Required columns for validation
_REQUIRED_RAW = [
    "sku_id", "date", "demand", "sales_quantity", "inventory_level",
    "lead_time", "unit_cost", "unit_price", "order_received"
]
_REQUIRED_COLUMNS = {
    "inventory_processed": _REQUIRED_RAW,
    "sku_class"          : ["sku_id", "class"],
    "sku_metric"         : ["sku_id", "fill_rate", "lost_sales", "DOI", "CV"],
    "forecast"           : ["sku_id", "date", "demand", "forecast"],
    "policy_sku"         : ["sku_id", "class", "safety_stock", "rop", "order_quantity"],
    "old_policy_metric"  : ["sku_id", "class", "fill_rate", "avg_inventory", "total_cost"],
    "new_policy_metric"  : ["sku_id", "class", "fill_rate", "avg_inventory", "total_cost"],
}


class DataLoader:
    def __init__(self, mode: str = "dashboard"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.mode   = mode

        self.inventory  = None
        self.sku_class  = None
        self.sku_metric = None
        self.forecast   = None
        self.policy_sku = None
        self.old_policy = None
        self.new_policy = None

    def load(self) -> "DataLoader":
        """Load data based on mode."""
        if self.mode == "pipeline":
            return self._load_pipeline()
        return self._load_dashboard()

    def _load_pipeline(self) -> "DataLoader":
        try:
            raw_path = os.path.join(_RAW_DIR, "inventory.csv")
            df = pd.read_csv(raw_path, parse_dates=[_DATE_COL])
            self.logger.info(f"Raw data loaded: {df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            raise

        missing = [c for c in _REQUIRED_RAW if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in raw data: {missing}")

        df = self._preprocess(df)
        self.inventory = df

        self.save_processed(df)

        return self

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[_DATE_COL] = pd.to_datetime(df[_DATE_COL])
        df = df.drop_duplicates(subset=[_SKU_COL, _DATE_COL])
        df = df.dropna()
        df = df.sort_values([_SKU_COL, _DATE_COL]).reset_index(drop=True)
        self.logger.info(f"Preprocessed data: {df.shape}")
        return df

    def save_processed(self, df: pd.DataFrame, filename: str = "inventory_processed.csv") -> None:
        os.makedirs(_PRO_DIR, exist_ok=True)
        path = os.path.join(_PRO_DIR, filename)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved → {path}")

    def _load_dashboard(self) -> "DataLoader":
        try:
            self.inventory  = pd.read_csv(os.path.join(_PRO_DIR, "inventory_processed.csv"), parse_dates=[_DATE_COL])
            self.sku_class  = pd.read_csv(os.path.join(_PRO_DIR, "sku_class.csv"))
            self.sku_metric = pd.read_csv(os.path.join(_PRO_DIR, "sku_metric.csv"))
            self.forecast   = pd.read_csv(os.path.join(_PRO_DIR, "forecast.csv"), parse_dates=[_DATE_COL])
            self.policy_sku = pd.read_csv(os.path.join(_PRO_DIR, "policy_sku.csv"))
            self.old_policy = pd.read_csv(os.path.join(_PRO_DIR, "old_policy_metric.csv"))
            self.new_policy = pd.read_csv(os.path.join(_PRO_DIR, "new_policy_metric.csv"))
            self.logger.info("Dashboard data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading dashboard data: {e}")
            raise

        if not self.validate():
            raise ValueError("Data validation failed — check logs for details.")

        return self

    def validate(self) -> bool:
        mapping = {
            "inventory_processed": self.inventory,
            "sku_class"          : self.sku_class,
            "sku_metric"         : self.sku_metric,
            "forecast"           : self.forecast,
            "policy_sku"         : self.policy_sku,
            "old_policy_metric"  : self.old_policy,
            "new_policy_metric"  : self.new_policy,
        }
        valid = True
        for name, df in mapping.items():
            if df is None:
                self.logger.error(f"{name} is not loaded.")
                valid = False
                continue
            missing = [c for c in _REQUIRED_COLUMNS[name] if c not in df.columns]
            if missing:
                self.logger.error(f"Missing columns in {name}: {missing}")
                valid = False
        return valid

    def get_class_summary(self) -> pd.DataFrame:
        df = self.sku_metric.merge(self.sku_class, on=_SKU_COL, how="left")
        return (
            df.groupby("class")
            .agg(
                sku_count     =(_SKU_COL,     "count"),
                avg_fill_rate =("fill_rate",  "mean"),
                avg_lost_sales=("lost_sales", "mean"),
                avg_doi       =("DOI",        "mean"),
            )
            .round(3)
            .reset_index()
        )

    def get_forecast_by_sku(self, sku_id: str) -> pd.DataFrame:
        return (
            self.forecast[self.forecast[_SKU_COL] == sku_id]
            .sort_values(_DATE_COL)
            .reset_index(drop=True)
        )

    def get_policy_comparison(self) -> pd.DataFrame:
        compare_cols = ["fill_rate", "avg_inventory", "holding_cost",
                        "ordering_cost", "stockout_cost", "total_cost"]
        old = self.old_policy.groupby("class")[compare_cols].mean().add_prefix("old_")
        new = self.new_policy.groupby("class")[compare_cols].mean().add_prefix("new_")
        return old.join(new).reset_index()

    def get_inventory_trend(self) -> pd.DataFrame:
        return (
            self.inventory.groupby(_DATE_COL)["inventory_level"]
            .sum()
            .reset_index()
            .rename(columns={"inventory_level": "total_inventory"})
        )

    def get_policy_sku(self, sku_id: str = None) -> pd.DataFrame:
        if sku_id:
            return self.policy_sku[self.policy_sku[_SKU_COL] == sku_id].reset_index(drop=True)
        return self.policy_sku

    def get_skus_by_class(self, cls: str) -> list:
        return self.sku_class[self.sku_class["class"] == cls][_SKU_COL].tolist()