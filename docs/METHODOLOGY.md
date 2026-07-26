# Methodology and decision contract

This document explains the decisions that are easy to misread from code alone.
The backend remains the executable source of truth; notebooks and the dashboard
are presentation layers over the same artifacts.

## 1. Time windows

The pipeline uses three disjoint predeclared windows:

| Window | Permitted use | Prohibited use |
| --- | --- | --- |
| Model selection (2025-01-01 to 2025-05-30) | Freeze ABC-XYZ/scope; compare forecast candidates and hyperparameters. | Policy calibration and final reporting. |
| Policy calibration (2025-05-31 to 2025-09-30) | Estimate forecast-error distributions and choose policy candidates after the forecasting method is frozen. | Retuning or reselecting the forecast model. |
| Locked evaluation (2025-10-01 to 2025-12-31) | Compare frozen policies on a shared demand path. | Any model, scope, parameter, or policy selection. |

Exact dates live in `src/backend/config/config.json`. The original
`validation_start`/`validation_end` keys remain for compatibility, but new code
uses the explicit selection and calibration boundaries.

The locked evaluation has already been reviewed during project development.
Results are therefore retrospective counterfactual evidence, not a pristine
prospective test and not a causal estimate of realized savings.

## 2. Demand and ABC value

`demand` is latent unconstrained unit demand. It is known in this synthetic case
study, while a production implementation would have to estimate demand censored
by stockouts.

ABC uses the economic basis selected by `abc_value_basis`:

- `gross_margin`: demand × (unit price − unit cost), the default because the
  scenario cost treats a lost unit as lost gross margin;
- `annual_consumption_cost`: demand × unit cost, suitable for working-capital
  prioritization;
- `revenue`: demand × unit price.

XYZ is a relative forecastability ranking based on causal model-selection error.
It is not an absolute claim that a SKU is intrinsically predictable.

## 3. Forecast contract

The displayed daily forecast is rolling one-day-ahead: the prediction for date
`t` may use actual demand only through `t-1`. Every target-derived lag, rolling
mean, rolling standard deviation, and exponentially weighted feature is shifted
before it is calculated.

Model candidates are compared on common calendar-date folds. Policy-calibration
forecasts are produced later with the selected method and hyperparameters
already frozen.

For a daily-review replenishment decision, an order placed after demand on date
`t` may use the stored forecast for `t+1`: demand on `t` is then already known,
so this alignment is causal. The simulator uses that forecast to update the
expected lead-time demand component of ROP. This is a dynamic level
approximation, not a complete probabilistic multi-horizon forecast; the
limitation must remain visible in reporting.

## 4. Policy and simulator contract

The simulator is a **daily-review approximation** of an `(s,Q)` policy:

1. receive due purchase orders at the start of the day;
2. fulfill demand up to available on-hand inventory;
3. record unfulfilled demand as lost sales;
4. calculate inventory position, including outstanding orders;
5. place a constrained order when inventory position is at or below ROP.

Lead time must be a positive whole number of days. Input data must contain a
complete daily calendar for each SKU; otherwise row-based lags and lead-time
windows are not calendar-day quantities.

The recorded raw `reorder_point`, `safety_stock`, and `order_quantity` are the
primary historical-policy baseline. A receipt-inferred ROP is retained only as
a sensitivity case because reconstructing order placement from receipts adds
measurement error.

For each in-scope SKU, the proposed policy evaluates the configured Cartesian
grid of cumulative-residual quantiles and order-quantity multipliers on the
policy-calibration window. Candidate Q is rounded up to its order multiple after
enforcing MOQ. If SKU-level `moq` or `order_multiple` fields are absent, the
documented scalar defaults are used.

Candidates that meet the intervention-specific fill-rate floor are ranked by
modeled total cost, then inventory. If no candidate in the finite grid meets the
floor, the highest-service candidate is retained only as an exception proposal
and `selection_status` is set to
`no_feasible_candidate_best_service`. The current run flags 11 of 109 in-scope
SKUs for SLA/grid review; it does not silently present them as feasible.

Both policies use the same evaluation demand path and initial on-hand state.
The current dataset does not expose open purchase orders at the evaluation
boundary, so both start with an empty pipeline. This is a documented limitation,
not evidence that real operations have no outstanding orders.

## 5. Metric definitions

- **Fill rate** = fulfilled units / demanded units.
- **Receipt-cycle service proxy** = receipt-defined cycles without lost units /
  receipt-defined cycles. It is a simulator diagnostic, not the same as fill
  rate.
- **Lead-time demand coverage target** = quantile used to calibrate cumulative
  lead-time forecast error. It is neither of the two realized service metrics
  above and must not be drawn as a direct fill-rate target.
- **DOI** = average on-hand units / average daily unit demand.
- **Unit turnover** = annualized portfolio unit demand / portfolio average
  on-hand units.
- **Value turnover** = annualized cost of demanded units / average inventory
  value.
- **Modeled total cost** = holding cost + ordering cost + modeled shortage cost.

The shortage-cost default is lost units × unit gross margin. Holding rate,
ordering cost, and shortage multiplier are assumptions, so the point estimate
must always be accompanied by assumption sensitivity.

## 6. Interpretation of uncertainty

The paired-SKU bootstrap measures sensitivity to which SKUs represent the
in-scope portfolio. It does not represent future demand, lead-time, cost, or
causal uncertainty.

Cost-assumption sensitivity varies holding rate, ordering cost, and shortage
multiplier across 36 configured cases.

The future stress layer creates 100 paired scenarios. For every in-scope SKU it
adds a seven-day moving-block bootstrap of dedicated calibration residuals to
the daily final forecast vector, clips negative demand to zero, and applies one
explicit scenario-level lead-time multiplier to both policies. The recorded
policy retains a static ROP; the proposed policy continues to use its causal
dynamic `t+1` forecast ROP. Both receive the same demand, starting inventory,
and lead-time shock. These are sensitivity scenarios, not calibrated prediction
intervals.

No interval in this project turns a synthetic counterfactual result into
realized savings.

## 7. Reproduce the project

Python 3.12 is recommended. Run the following PowerShell commands from the
repository root.

Create an isolated environment and install the runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install notebook dependencies only when you intend to run Jupyter:

```powershell
python -m pip install -r requirements-notebooks.txt
```

Regenerate all backend artifacts:

```powershell
python src/backend/main.py
```

Run the regression suite:

```powershell
python -m unittest discover -s tests -v
```

Launch the dashboard:

```powershell
streamlit run src/frontend/app.py
```

## 8. Production extensions

A production implementation would additionally require:

- current on-hand, open purchase orders, backorders, and inventory adjustments.
- governed SKU-level MOQ/order-multiple master data, supplier capacity and
  calendars, and empirical lead-time distributions.
- promotion, lifecycle, substitution, location, and channel signals.
- approval workflow and export to the replenishment/ERP system.
- monitoring for forecast value added, service, working capital, overrides, drift, and realized policy performance.