"""
Phase H - Demand Forecasting
Aggregates weekly order quantity, builds a naive baseline and a
Holt-Winters Exponential Smoothing model with a time-aware train/test
split, evaluates with MAE/RMSE, and plots actual vs forecast.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

df = pd.read_csv("data/processed/cleaned_supply_chain.csv", parse_dates=["order date (DateOrders)"])

# Weekly aggregate of total quantity demanded (company-wide)
weekly = (df.set_index("order date (DateOrders)")
            .resample("W")["Order Item Quantity"].sum())
weekly = weekly[weekly.index < weekly.index.max()]  # drop partial last week

# DATA QUALITY FINDING: from 2017-10-08 onward, weekly volume flatlines to a
# suspiciously constant ~479-480 units/week (vs organic 400-2600 range before
# it) - a known artifact of this Kaggle release, not genuine demand. Using it
# would make every model look artificially perfect. It is excluded here and
# the finding is documented in the final report.
CUTOFF = "2017-10-01"
anomaly_tail = weekly[weekly.index > CUTOFF]
weekly = weekly[weekly.index <= CUTOFF]

# Time-aware split: last 12 weeks as test set
test_size = 12
train, test = weekly.iloc[:-test_size], weekly.iloc[-test_size:]

# 1. Naive baseline: last observed value repeated
naive_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)

# 2. Moving-average baseline (4-week rolling mean of train, held constant)
ma_pred = pd.Series([train.tail(4).mean()] * len(test), index=test.index)

# 3. Holt-Winters Exponential Smoothing (trend + weekly seasonality)
hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=52 if len(train) > 104 else 13,
                                 initialization_method="estimated").fit()
hw_pred = hw_model.forecast(test_size)

def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)}

results = [
    eval_model("Naive (last value)", test, naive_pred),
    eval_model("Moving Average (4wk)", test, ma_pred),
    eval_model("Holt-Winters Exp. Smoothing", test, hw_pred),
]

# Plot actual vs forecasts
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(weekly.index, weekly.values, label="Actual (full history)", color="#333333", alpha=0.5)
ax.plot(test.index, test.values, label="Actual (test period)", color="black", linewidth=2)
ax.plot(test.index, naive_pred.values, label="Naive baseline", linestyle="--", color="#C1443C")
ax.plot(test.index, ma_pred.values, label="Moving avg baseline", linestyle="--", color="#E07B39")
ax.plot(test.index, hw_pred.values, label="Holt-Winters forecast", linestyle="-", color="#2E5EAA", linewidth=2)
ax.set_title("Weekly Demand: Actual vs Forecast (last 12 weeks held out)")
ax.set_xlabel("Week"); ax.set_ylabel("Units Ordered")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/project/reports/figures/11_demand_forecast.png", dpi=130)
plt.close(fig)

best_model = min(results, key=lambda r: r["RMSE"])

# Simple planning recommendation from the forecast
next_4wk_avg = hw_model.forecast(4).mean()
recent_4wk_avg = train.tail(4).mean()
pct_change = round(100 * (next_4wk_avg - recent_4wk_avg) / recent_4wk_avg, 1)
recommendation = (
    f"Forecasted demand for the next 4 weeks averages {next_4wk_avg:.0f} units/week, "
    f"a {'increase' if pct_change >= 0 else 'decrease'} of {abs(pct_change)}% vs the most "
    f"recent 4-week average ({recent_4wk_avg:.0f} units/week). "
    + ("Recommend increasing replenishment buffers for fast-moving SKUs ahead of this uptick."
       if pct_change > 5 else
       "Recommend holding replenishment at current levels; no material demand shift forecast."
       if abs(pct_change) <= 5 else
       "Recommend tapering incoming replenishment to avoid excess stock as demand softens.")
)

output = {
    "data_quality_finding": (
        f"Weekly order volume flatlines to a constant ~479-480 units from "
        f"{anomaly_tail.index.min().date()} to {anomaly_tail.index.max().date()} "
        f"({len(anomaly_tail)} weeks) - inconsistent with the organic 400-2600/week "
        f"variability seen earlier in the series. Treated as a known synthetic-tail "
        f"artifact of this dataset release and excluded from model training/testing "
        f"so evaluation metrics reflect genuine demand patterns."
    ),
    "usable_history_range": f"{weekly.index.min().date()} to {weekly.index.max().date()}",
    "test_period_weeks": test_size,
    "model_comparison": results,
    "best_model_by_rmse": best_model["model"],
    "next_4_week_forecast_avg_units": round(next_4wk_avg, 1),
    "planning_recommendation": recommendation,
}
with open("/home/claude/project/reports/phaseH_forecasting_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(json.dumps(output, indent=2, default=str))
