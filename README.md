# Infotact Solutions-Project-2--Supply-Chain-Analytics-Demand-Forecasting-Anomaly-Detection-
---

## Week 1 — Time-Series Preprocessing & Decomposition ✅

**Script:** `Week1_TimeSeries_Preprocessing.py`

### Datasets Loaded
| File | Rows | Date Range |
|------|------|-----------|
| sales_data.csv | 76,000 | 2022-01-01 → 2024-01-30 |
| retail_store_inventory.csv | 73,100 | 2022-01-01 → 2024-01-01 |
| demand_forecasting_dataset.csv | 4,999 | 2023-01-01 → 2036-09-07 |

### Steps Performed
- Loaded all 3 datasets using Python Pandas with parse_dates
- Standardised column names to snake_case
- Confirmed zero missing values across 154,099 total rows
- Aggregated daily data by category across all stores and regions
- Created complete date range index and filled gaps via time interpolation
- Resampled: Daily → Weekly (110 weeks) → Monthly (25 months)
- Applied statsmodels seasonal_decompose (additive, period=52 weeks)

### Decomposition Results
| Category | Avg Daily | Trend Range | Seasonal Amplitude |
|---|---|---|---|
| Electronics | 996 units | 6,721 → 7,120 | 4,918 |
| Groceries | 4,115 units | 28,057 → 29,420 | 18,641 |
| Clothing | 1,514 units | 10,377 → 10,648 | 3,316 |
| Furniture | 1,159 units | 7,756 → 8,349 | 5,877 |
| Toys | 1,098 units | 7,189 → 7,955 | 8,292 |

### Key Finding
Toys has the highest seasonal amplitude (8,292) relative to its
trend (7,189–7,955) — meaning its demand swings most dramatically
across seasons, making it the highest priority category for
accurate demand forecasting in Week 3.
Holiday season and promotion flags both measurably increase
sales_units — these will be used as external regressors in
the Prophet forecasting model in Week 3.
