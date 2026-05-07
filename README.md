# Inventory Optimization

## Objective
- Analyze SKU demand patterns and segment products to improve inventory decisions.  
- Focus on aligning inventory policies with demand behavior.

## Dataset
- Inventory dataset (sku_id, date, sales_quantity, inventory_level, rop, order_quantity,...)

## Project Structure
```
inventory-optimization/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_ABC_XYZ.ipynb
├── .gitignore
└── README.md
```

## Progress
- [x] Data preprocessing  
- [x] Exploratory Data Analysis (EDA)  
- [x] ABC–XYZ segmentation  
- [ ] Forecasting  
- [ ] Inventory policy simulation 

## Key Findings
- Inventory performance is uneven across SKUs, with a small subset contributing disproportionately to lost sales.  
- High-value A-class SKUs experience lower fill rates and higher lost sales, indicating understocking and the need for improved forecasting and safety stock policies.  
- Several low-value C-class SKUs maintain excessive inventory coverage, particularly CZ items, suggesting potential overstock and inefficient capital allocation.  
- Differences in demand variability across XYZ categories highlight the need for differentiated inventory policies rather than a uniform replenishment strategy.

## Next Steps
- Forecasting
- Optimize inventory policies:
  - Safety stock  
  - Reorder point (ROP)  
  - Order quantity (EOQ)  
- Simulate trade-offs between service level and inventory cost