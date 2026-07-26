# Inventory Policy Optimization

An end-to-end supply chain analytics case study for a synthetic 150-SKU
portfolio. The project classifies inventory, evaluates demand forecasts,
calibrates reorder policies, compares policy outcomes, and produces an
actionable Streamlit dashboard.

All financial values are modeled outputs on synthetic data. They are decision
support estimates, not realized or causal business savings.

## Business question

The project answers three practical questions:

1. Which SKUs need a policy change?
2. What reorder point and order quantity should be proposed for those SKUs?
3. Does the proposed policy improve service, inventory, and modeled cost under
   the same demand conditions as the current policy?

## Key results

### Optimization scope

The pipeline selected 109 of 150 SKUs for policy optimization before policy
calibration began. Across the historical data through 2025-09-30, those 109
SKUs represent:

- 72.7% of portfolio SKUs;
- 74.1% of unit demand;
- 88.4% of gross-margin demand value.

### Full-portfolio policy comparison

This is the first table shown on the Inventory Policy page. Both scenarios
contain all 150 SKUs.

| Scenario | SKUs | Lost demand | Simulated fill rate | Avg portfolio on-hand | Modeled total cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current policy | 150 | 4,285 | 97.49% | 51,885 | $5,741,018 |
| Proposed policy (109 optimized + 41 unchanged) | 150 | 2,220 | 98.70% | 36,612 | $2,932,370 |

The proposed full-portfolio scenario applies the new policy to 109 optimized
SKUs and leaves the remaining 41 SKUs on their current policy.

### Impact on 109 optimized SKUs

These four rows match the KPI cards shown below the full-portfolio table on the
dashboard.

| KPI | Proposed outcome | Change vs current policy for the same 109 SKUs |
| --- | ---: | ---: |
| Simulated fill rate | 98.65% | +1.63 pts |
| Avg portfolio on-hand | 22,042 units | -40.93% |
| Modeled total cost | $2,497,106 | -52.94% |
| Lost demand | 1,706 | -54.76% |

The change column does not use the 150-SKU Current policy row from the first
table. It compares the proposed policy with the current-policy simulation for
the same 109 optimized SKUs.

### Stress-test results

The dashboard also tests the proposed policy across 100 paired demand and lead
time scenarios.

| Stress-test KPI | Median | 90% scenario range |
| --- | ---: | ---: |
| Fill-rate change | +1.82 pts | +0.88 to +4.74 pts |
| Inventory change | -41.10% | -43.05% to -35.20% |
| Total-cost change | -60.57% | -65.50% to -52.21% |
| Scenarios with lower modeled cost | 100.0% | — |

Modeled savings are also positive in all 36 configured cost-assumption cases.
11 of 109 optimized SKUs do not reach the required calibration fill rate in the
tested policy grid and are shown as `Planner review`.

## How to interpret the policy comparison

The dashboard deliberately separates actual historical performance from
simulated policy performance:

| Dashboard term | Meaning |
| --- | --- |
| Historical actual fill rate | Calculated directly from raw sales divided by demand for all 150 SKUs through 2025-09-30. |
| Current policy | Simulation using each SKU's recorded pre-period reorder point, order quantity, and lead time. |
| Proposed policy | Simulation using the calibrated policy for 109 SKUs while 41 SKUs remain unchanged. |

The Current policy and Proposed policy simulations start with the same
inventory and receive the same demand from 2025-10-01 to 2025-12-31. Running
both policies through the same simulator creates a like-for-like comparison.

The 95.51% fill rate on the Overview page is an actual historical KPI. It should
not be compared directly with the simulated fill rates on the Inventory Policy
page.

## Data and evaluation design

The fixed synthetic input contains 109,650 daily rows:

- 150 SKUs;
- one row per SKU per calendar day;
- date range from 2024-01-01 to 2025-12-31;
- `demand` represents unconstrained demand;
- `sales_quantity` represents fulfilled demand;
- `inventory_level` represents end-of-day on-hand inventory.

The analytical process uses three non-overlapping windows:

| Window | Dates | Purpose |
| --- | --- | --- |
| Model selection | 2025-01-01 to 2025-05-30 | Compare and tune forecast methods; freeze ABC-XYZ and optimization scope. |
| Policy calibration | 2025-05-31 to 2025-09-30 | Estimate forecast risk and select policy parameters. |
| Locked evaluation | 2025-10-01 to 2025-12-31 | Compare the frozen current and proposed policies once. |

Separating these windows prevents the final evaluation period from influencing
model selection or policy calibration.

## Analytical workflow

### 1. ABC-XYZ classification

ABC ranks SKUs by historical gross-margin contribution:

```text
demand × (unit_price - unit_cost)
```

XYZ ranks SKUs by relative one-day-ahead forecast error during the
model-selection window. The resulting classes support three intervention
groups:

| Classes | Intervention | Decision intent |
| --- | --- | --- |
| AX, AY | Protect strategic value | Maintain service for high-value demand |
| CX | Correct understock | Recover service where lost demand is material |
| AZ, BZ, CZ | Reduce overstock | Lower inventory while preserving service |

The generated scope is saved in
`data/processed/optimization_scope.csv`; the dashboard reads that artifact
instead of maintaining a second class list.

### 2. Demand forecasting

Forecasts are rolling one-day-ahead, so the forecast for date `t` can only use
actual demand through `t-1`.

The current pipeline run selected:

- LightGBM for AX and AY;
- Naive for CX;
- Historic Mean for AZ, BZ, and CZ.

Portfolio WAPE for the 109-SKU locked evaluation scope is 33.52%. MAE, RMSE,
Bias, MASE, and RMSSE are also available at portfolio, class, and SKU levels.

### 3. Policy calibration

For each in-scope SKU, the pipeline:

- estimates safety stock from empirical cumulative lead-time forecast errors;
- tests a grid of safety-stock quantiles and order-quantity multipliers;
- respects minimum order quantity and order multiple inputs;
- selects the lowest modeled-cost candidate that meets the required calibration
  fill rate;
- flags the best-service candidate for planner review when no tested candidate
  is feasible.

The proposed policy uses the next-day forecast in the reorder-point decision:

```text
ROP today = next-day forecast × lead time + safety stock
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for simulator timing, formulas,
assumptions, and metric definitions.

## Dashboard

[Open the deployed Streamlit dashboard](https://inventory-policy-optimization.streamlit.app)

The Streamlit dashboard contains four pages:

- **Overview:** historical performance for all 150 SKUs, inventory trend,
  inventory by class, and slow-moving or stockout-risk SKU alerts.
- **ABC-XYZ Analysis:** classification matrix, historical class metrics, and
  policy optimization priorities.
- **Forecast Performance:** portfolio forecast KPIs and individual SKU
  actual-versus-forecast analysis.
- **Inventory Policy:** full-portfolio comparison, impact on 109 optimized
  SKUs, 100-scenario stress test, sensitivity details, class summary, and
  downloadable SKU actions.

The dashboard reads only canonical pipeline artifacts and validates their hashes
against `data/metadata/artifact_manifest.json`.

## Main outputs

| Artifact | Purpose |
| --- | --- |
| `data/processed/optimization_scope.csv` | Selected classes and SKU coverage |
| `data/processed/validation_forecast.csv` | Policy-calibration forecasts |
| `data/processed/forecast.csv` | Locked-evaluation forecasts |
| `data/processed/forecast_metrics.csv` | Portfolio, class, and SKU forecast metrics |
| `data/processed/policy_candidate_audit.csv` | Every tested policy candidate and feasibility result |
| `data/processed/policy_sku.csv` | Selected policy parameters by SKU |
| `data/processed/old_policy_metric.csv` | Current-policy simulation for 109 optimized SKUs |
| `data/processed/new_policy_metric.csv` | Proposed-policy simulation for 109 optimized SKUs |
| `data/processed/full_policy_summary.csv` | Full-portfolio comparison for all 150 SKUs |
| `data/processed/policy_action.csv` | Prioritized implementation actions by SKU |
| `data/processed/policy_uncertainty.csv` | Paired-SKU bootstrap sensitivity |
| `data/processed/policy_sensitivity.csv` | Cost-assumption sensitivity cases |
| `data/processed/historical_policy_sensitivity.csv` | Current-policy ROP sensitivity: recorded versus receipt-inferred |
| `data/processed/scenario_uncertainty.csv` | Demand and lead-time stress scenarios |
| `data/metadata/artifact_manifest.json` | Data windows, models, assumptions, hashes, and runtime lineage |

The five notebooks are read-only walkthroughs of these canonical artifacts. They
do not create a separate set of production outputs.

## Run locally

Python 3.12 is recommended. Run every command from the repository root.

```powershell
# 1. Create and activate a virtual environment.
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the runtime dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Rebuild all processed artifacts and the manifest.
python src/backend/main.py

# 4. Run the regression tests.
python -m unittest discover -s tests -v

# 5. Start the dashboard.
streamlit run src/frontend/app.py
```

The full pipeline can take several minutes because it includes forecast model
selection, policy candidate evaluation, bootstrap analysis, sensitivity cases,
and stress scenarios.

Install notebook dependencies only when needed:

```powershell
python -m pip install -r requirements-notebooks.txt
jupyter lab notebooks
```

Rebuild the pipeline after changing raw data, backend logic, or configuration.
A frontend-only change does not require rebuilding the analytical artifacts.

## Quality controls

The regression suite checks:

- unique SKU-date keys and complete daily calendars;
- leakage-safe feature construction and time folds;
- disjoint model-selection, calibration, and evaluation windows;
- identical starting inventory and demand paths for policy comparison;
- inventory conservation and lead-time timing;
- non-negative forecast outputs;
- paired and deterministic sensitivity calculations;
- manifest membership, file sizes, and SHA-256 hashes;
- notebook use of canonical read-only artifacts.

## Repository structure

```text
inventory-optimization/
|-- data/
|   |-- raw/                 # Fixed synthetic pipeline input
|   |-- processed/           # Canonical forecast and policy outputs
|   `-- metadata/            # Data contract and artifact manifest
|-- docs/                    # Methodology and interpretation guidance
|-- notebooks/               # Read-only analytical walkthroughs
|-- params/                  # Model parameters and feature importance
|-- src/
|   |-- backend/             # Classification, forecasting, simulation, pipeline
|   `-- frontend/app.py      # Streamlit dashboard
|-- tests/                   # Data, leakage, simulator, decision, and lineage tests
|-- requirements.txt
`-- requirements-notebooks.txt
```

## Limitations

- The dataset is synthetic and cannot establish external validity.
- Unconstrained demand is observable only because the data are synthetic; real
  sales data would require stockout-censoring treatment.
- Dollar amounts are modeled cost outputs, not accounting or realized savings.
- The simulator starts from pre-period on-hand inventory and does not reconstruct
  open purchase orders at the evaluation boundary.
- Supplier capacity, multi-echelon effects, substitutions, shelf life, and
  negotiated commercial constraints are outside the model.
- Stress tests are sensitivity scenarios, not calibrated prediction intervals.
- The locked evaluation period has already been reviewed; production validation
  requires a new untouched period or a prospective pilot.

## Author

- GitHub: https://github.com/wuandt
- LinkedIn: https://www.linkedin.com/in/trvquan