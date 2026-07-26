# Data layout

| Directory | Purpose | Used by |
| --- | --- | --- |
| `raw/` | Immutable source input. | Pipeline preprocessing. |
| `processed/` | Canonical CSV artifacts consumed by the dashboard and downstream pipeline stages. | Backend, dashboard, and notebooks. |
| `metadata/` | JSON metadata: the inventory data contract and the artifact manifest with content hashes. | Backend, dashboard, notebooks, and tests. |

Directory names in the table are relative to `data/`.

The project treats `raw/inventory.csv` as a fixed synthetic case-study input.
Upstream data generation is outside the analytical scope; reproducibility starts
from this versioned raw artifact.

`raw/inventory.csv` and `processed/inventory_processed.csv` can be byte-identical when validation and preprocessing do not alter any records. They remain separate because the first is the immutable source while the second is the validated dashboard artifact.

Do not delete files from `metadata/`: `inventory_data_contract.json` documents the processed-data schema and `artifact_manifest.json` detects missing or stale artifacts before the dashboard loads.

The analytical notebooks are read-only walkthroughs of the canonical
`processed/` artifacts. They do not maintain a second set of CSV outputs; this
prevents notebook and production logic from silently drifting apart.

## Business semantics

- Grain: one row per SKU per calendar day.
- `demand`: latent unconstrained unit demand. This is observable only because the
  case study is synthetic; real sales data would require stockout-censoring
  treatment before it could be used as unconstrained demand.
- `sales_quantity`: fulfilled demand, capped by available inventory.
- `inventory_level`: end-of-day on-hand units.
- `order_received`: units received at the start of the day.
- `reorder_point`, `safety_stock`, and `order_quantity`: recorded parameters of
  the historical policy. They are the primary simulation baseline.
- `unit_cost` and `unit_price`: scenario currency per unit. The currency is
  intentionally generic; monetary results are modeled cost values, not realized
  accounting savings.

All rows in the published dataset are daily and complete. Production ingestion
must either enforce the same calendar grid or reindex missing dates explicitly;
row-based lags cannot safely be interpreted as calendar-day lags otherwise.