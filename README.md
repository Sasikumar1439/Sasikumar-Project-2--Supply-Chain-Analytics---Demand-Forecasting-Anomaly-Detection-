# Infotact Solutions-Project-2--Supply-Chain-Analytics-Demand-Forecasting-Anomaly-Detection-
---
## Data sets:
"C:\Users\Vinay\Downloads\demand_forecasting_dataset.csv",
          "C:\Users\Vinay\Downloads\Supply_Chain_Analytics_project\sales_data.csv",
          "C:\Users\Vinay\Downloads\retail_store_inventory.csv".

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

---

## Week 2 — Statistical Anomaly Detection ✅

**Script:** `Week2_Anomaly_Detection.py`

### Methods Implemented
| Method | Anomalies Found | Rate | Recommended For |
|--------|----------------|------|-----------------|
| Z-Score (threshold ±2.5) | 0 at aggregate | 0% | Symmetric distributions |
| IQR (1.5 × IQR fence) | 122 | 16% | Skewed distributions |
| Isolation Forest (5%) | 38 | 5% | Multi-dimensional patterns |
| Confirmed (2+ methods) | 19 | 2.5% | Production threshold |

### Steps Performed
- Loaded sales_data.csv and retail_store_inventory.csv
- Built daily aggregated time series for units_sold and inventory_level
- Extracted weather_condition and seasonality per day using mode()
- Ran Z-Score, IQR and Isolation Forest anomaly detection
- Confirmed anomalies where 2 or more methods agreed
- Contextualized each anomaly using Epidemic, Promotion,
  Weather and Seasonality columns
- Classified anomaly types: Epidemic Spike/Drop,
  Promotion Spike, Weather Drop, Unexplained

### Key Findings
- 94.7% of confirmed anomalies occurred during Epidemic days
  (Epidemic column is near-perfect ground truth predictor)
- IQR method alone flags 16% of days — too many false positives
- Isolation Forest at 5% is the recommended default for Streamlit app
- Clothing category had the most individual anomalies
- Forecast error and anomaly days overlap — anomalies directly
  reduce forecast accuracy (motivation for Week 3 Prophet model)

### Output Files
| File | Description |
|------|-------------|
| Week2_Anomaly_Dashboard.png | 6-panel anomaly detection dashboard |
| Week2_PerCategory_Anomalies.png | Per-category Z-Score anomaly charts |
| Week2_Inventory_Anomalies.png | Inventory level + forecast vs actual |
| anomaly_results.csv | All confirmed anomalies with context |
| anomaly_method_comparison.csv | Method benchmark comparison |
| sql/anomaly_detection_queries.sql | SQL Z-Score + IQR + context queries |
