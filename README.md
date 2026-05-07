# Inventory Optimization

## Objective
- Analyze SKU demand patterns and segment products to improve inventory decisions.  
- Focus on aligning inventory policies with demand behavior.

## Dataset
- Inventory dataset (sku_id, date, sales_quantity, inventory_level, rop, order_quantity,...)

## Project Structure
```
inventory-optimization/
├── .vscode
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_ABC_XYZ.ipynb
│   └── 04_forecasting.ipynb
├── params/
├── .gitignore
└── README.md
```

## Progress
- [x] Data preprocessing  
- [x] Exploratory Data Analysis (EDA)  
- [x] ABC–XYZ segmentation  
- [x] Forecasting  
- [ ] Inventory policy simulation 

## Key Findings
- **Significant Performance Improvement over Baselines**:  
  The optimized LightGBM model delivers substantially higher forecasting accuracy, achieving an **MAE of 4.74**. This represents approximately a **30% improvement** compared to the Naive model (MAE = 6.82) and Seasonal Naive model (MAE = 6.62).
- **Superior Bias Correction**:  
  While traditional baseline models suffer from severe systematic under-forecasting — with %Bias of **-15.69%** (Naive) and **-9.82%** (Seasonal Naive) — the LightGBM model successfully reduces the bias to a near-ideal **-2.41%**.  
  This significant improvement is critical in minimizing stock-out risks and maintaining high service levels.
- **Acceptable Volatility Handling for Safety Stock Calculation**:  
  Although the %MAE (~29.9%) and %RMSE (~39.1%) remain relatively high due to the volatile nature of electronics demand at the SKU level, these error metrics are still usable. In particular, the %RMSE can be effectively leveraged to calculate safety stock, enabling a balanced approach between service level and inventory cost.

## Next Steps
- Optimize inventory policies:
  - Safety stock  
  - Reorder point (ROP)  
  - Order quantity (EOQ)  
- Simulate trade-offs between service level and inventory cost