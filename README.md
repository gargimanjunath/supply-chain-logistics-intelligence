# Supply Chain & Logistics Intelligence System

End-to-end Data Analytics / Data Science capstone project built on the
**DataCo Smart Supply Chain** dataset (180,519 order-line-item records,
Jan 2015 - Jan 2018), spanning data engineering, SQL analysis, statistical
testing, demand forecasting, and machine learning classification.

## Results Summary

| Metric | Value |
|---|---|
| Total Revenue | $36,784,735.01 |
| Total Orders (distinct) | 65,752 |
| Average Order Value | $559.45 |
| On-Time Delivery Rate | 45.17% |
| Late-Delivery Rate | 54.83% |
| Shipping mode -> delivery time (ANOVA) | p < 0.001 (significant) |
| Customer segment -> lateness (Chi-square) | p = 0.599 (not significant) |
| Forecast model | Holt-Winters (29.5% MAE improvement over naive) |
| Best late-delivery classifier | Random Forest (ROC-AUC 0.75) |

## Project Structure
```
supply-chain-logistics-project/
├── data/
│   ├── raw/                  # original, untouched CSVs
│   └── processed/            # cleaned parquet/CSV + SQLite DB
├── sql/                      # 12 MySQL-syntax business queries
├── src/
│   ├── preprocessing/        # Phase A cleaning
│   ├── analytics/            # Phase B-D-F-G: SQL runner, KPIs, EDA, inventory, stats
│   ├── forecasting/          # Phase H: Holt-Winters demand forecast
│   ├── prediction/           # Phase I: 4-model late-delivery classifier
│   └── genai_executive_assistant.py
├── models/                   # saved best ML model (.joblib)
├── dashboard/                # Power BI-ready Excel extract + build guide
├── api/                      # FastAPI service (predict-delay, forecast, product, region)
├── reports/                  # all JSON/CSV results, 14+ EDA charts, final PDF
├── logs/                     # data-cleaning decision log
└── requirements.txt
```

## How to Reproduce
```bash
pip install -r requirements.txt

python src/preprocessing/clean_data.py          # Phase A
python src/analytics/run_sql_queries.py         # Phase B
python src/analytics/kpis.py                    # Phase C
python src/analytics/eda_charts.py              # Phase D
python src/analytics/inventory_analytics.py     # Phase F
python src/analytics/statistics_tests.py        # Phase G
python src/forecasting/forecast_demand.py       # Phase H
python src/prediction/ml_late_delivery.py       # Phase I

# Optional
uvicorn api.main:app --reload --port 8000       # FastAPI service
python src/genai_executive_assistant.py         # needs ANTHROPIC_API_KEY
```

## Key Data-Quality Decisions
- **Grain**: one row = one order **line item**, not one order (65,752 unique
  orders across 180,519 line items). All order-level KPIs de-duplicate on
  `Order Id` first.
- **Forecast truncation artifact**: Oct 2017-Jan 2018 shows an abrupt volume
  cliff in the raw export that is a known Kaggle dataset artifact, not a real
  demand collapse (visible in `reports/figures/01_monthly_sales_trend.png`).
  These months are excluded from forecast model training.
- **No leakage in ML**: the late-delivery classifier uses only features
  knowable at order time (shipping mode, scheduled duration, region,
  category, order value, segment, timing). Post-fulfillment fields (actual
  shipping days, delivery status, realized profit) are excluded.
- **Documented proxies, not invented data**: the dataset has no dedicated
  supplier entity (Department Name is used as a labeled proxy, Phase E) and
  no on-hand inventory field (demand-based velocity/variability indicators
  are used instead of claimed stock levels, Phase F).

## Data Source
DataCo Smart Supply Chain for Big Data Analysis, Kaggle
(https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
