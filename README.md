# Inventory Optimization

## Objective
- Analyze SKU demand patterns and segment products to improve inventory decisions.  
- Focus on aligning inventory policies with demand behavior.

## Dataset
- Inventory dataset (sku_id, date, sales_quantity, inventory_level, rop, order_quantity,...)

## Project Structure
```
inventory-optimization/
├── .vscode/
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_ABC_XYZ.ipynb
│   ├── 04_forecasting.ipynb
│   └── 05_simulation.ipynb
├── params/
└── README.md
```

## Progress
- [x] Data preprocessing  
- [x] Exploratory Data Analysis (EDA)  
- [x] ABC–XYZ segmentation  
- [x] Forecasting  
- [X] Inventory policy simulation 

## Key Findings
- **Improved Service–Cost Trade-off**:  
  Fill rate increased **AX (+2.6%, 95.82% → 98.31%)**, **AY (+2.3%, 95.42% → 97.61%)** while total cost dropped significantly (**up to -56.1%**).

- **Lost Sales Significantly Reduced**:  
  CX stockout cost decreased **-82.8% (146.90 → 25.25)** with only a small inventory increase (**+7.8%**).

- **More Efficient Inventory Allocation**:  
  BZ, CZ inventory reduced **-19% to -26%** while maintaining high fill rate (**~98%+**), indicating a controlled cost–service trade-off.

- **Consistent Performance Across Segments**:  
  Policy improves both service and cost efficiency across A, B, and C classes.

**-> Overall:** Achieves a strong balance between service level and total cost, suitable for practical implementation.