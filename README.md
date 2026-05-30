# 📦 Inventory Optimization

## Dataset
The dataset was synthetically generated to simulate a realistic electronics retail inventory environment.
| Property | Value |
|---|---|
| SKUs | 150 |
| Time Period | 2024-01-01 → 2025-12-31 |
| Records | 109,650 rows |
| Product Categories | Electronics & Accessories |
| Policy | Continuous review policy (s, Q) |

---

## Problem Statement
The inventory system faces both understock and overstock issues across different SKU groups.
- **AX, AY** — High-value SKUs with fill rates below target levels, increasing stockout risk and lost sales
- **CX** — Understocked SKUs with insufficient inventory coverage and the highest lost sales
- **BZ, CZ** — Overstocked SKUs with excessive holding costs despite already high service levels

> The goal is to improve service levels while reducing unnecessary inventory and total inventory-related costs.

---

## Business Impact
-  **AX, AY** — Achieved the strongest performance, increasing fill rate by over 2% while reducing inventory by 6–11%, cutting stockout costs by more than 56%, and reducing total cost by approximately 50%.
- **CX** — Successfully corrected understock conditions. Fill rate improved from 92.56% to 95.70% while inventory decreased slightly. Stockout cost fell by 45.86%, resulting in a 39.1% reduction in total cost.
- **BZ, CZ** — Reduced excess inventory by 20–26% with only a minor decrease in fill rate. The increase in stockout cost was outweighed by savings in holding cost, leading to a more balanced inventory position.

> Overall, the optimized inventory policy improved service levels for understocked classes while reducing excess inventory in overstocked classes, resulting in a more cost-efficient and balanced inventory system.

---

## End-to-End Pipeline
```
Raw Inventory Data
        │
        ▼
Data Preprocessing & Feature Engineering
        │
        ▼
ABC-XYZ Analysis
        │
        ▼
Demand Forecasting
        │
        ▼
Inventory Policy Optimization
        │
        ▼
Policy Simulation & Evaluation
        │
        ▼
Interactive Streamlit Dashboard
```

---

## Interactive Dashboard
An interactive Streamlit dashboard to explore the full analysis results across 4 pages.

👉 **[Open Dashboard](https://inventory-policy-optimization.streamlit.app)**

| Page | Description |
|------|-------------|
| 🏠 Overview | Inventory health KPIs, inventory level trend, and alerts for slow-moving / stockout-risk SKUs |
| 📊 ABC-XYZ Analysis | SKU segmentation matrix and per-class diagnosis (understock, overstock, low service) |
| 📈 Forecast Performance | Actual vs forecast chart with accuracy metrics (%MAE, %RMSE, %Bias) per SKU |
| 🎯 Policy Comparison | Side-by-side old vs new policy metrics (fill rate, inventory, cost breakdown) per class |

---

## Project Structure
```
inventory-optimization/
├── data/                  # Raw & processed data
├── model/                 # Trained models (lgbm_model.pkl)
├── notebooks/             # 01_preprocessing → 05_simulation
├── params/                # Hyperparameters
├── src/
│   ├── backend/           # Data pipeline: load → feature engineering → forecast → simulation
│   └── frontend/app.py    # Streamlit dashboard
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Tech Stack
| Layer | Tools |
|---|---|
| Data Processing | pandas, numpy |
| Forecasting | LightGBM, statsforecast |
| Hyperparameter Tuning | optuna |
| Inventory Optimization | Safety Stock, Reorder Point (ROP), Order Quantity (Q) |
| Visualization | plotly, streamlit |
| Evaluation | scipy, scikit-learn |

---

## Getting Started
**1. Clone repository**
```bash
git clone https://github.com/your-username/inventory-optimization.git
cd inventory-optimization
```
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Run dashboard**
```bash
streamlit run src/frontend/app.py
```

---

## Key Takeaways
- Inventory policies should vary by SKU behavior and business importance
- Improving service level does not always require higher inventory
- Forecasting accuracy alone is insufficient without operational evaluation
- Policy simulation helps quantify trade-offs between availability and cost

---

## Author
**Your Name**
- GitHub: https://github.com/wuandt
- LinkedIn: https://www.linkedin.com/in/trvquan

---
