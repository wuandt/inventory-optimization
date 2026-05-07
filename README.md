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
├── .gitignore
└── README.md
```

## Progress
- [x] Data preprocessing  
- [ ] Exploratory Data Analysis (EDA)  
- [ ] ABC–XYZ segmentation  
- [ ] Forecasting  
- [ ] Inventory policy simulation 

## Current Focus
- Clean and structure transactional inventory data  
- Prepare dataset for time-series and SKU-level analysis 

## Next Steps
- Perform EDA to understand demand and inventory behavior
- Conduct ABC–XYZ segmentation
- Implement inventory policies (EOQ, ROP, safety stock)  
- Simulate cost vs service level trade-offs