"""
Phase H - Demand Forecasting.

Aggregates order quantity by month, excludes the known post-Oct-2017 data
truncation artifact (confirmed visually in EDA chart 01/13 - volume falls off
a cliff starting Oct 2017, which is a Kaggle export artifact, not a real
demand collapse), fits a naive baseline + Holt-Winters model on a time-aware
train/test split, and evaluates with MAE/RMSE.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

df = pd.read_parquet("data/processed/dataco_clean.parquet")

monthly = df.groupby("order_year_month").agg(
    quantity=("Order Item Quantity", "sum"),
    sales=("Sales", "sum"),
).sort_index()
monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()

# Exclude the truncated tail (Oct 2017 - Jan 2018): confirmed data artifact,
# not a genuine demand drop -- keeping it would teach the model a fake crash.
COMPLETE_CUTOFF = "2017-09-30"
monthly_complete = monthly.loc[:COMPLETE_CUTOFF].copy()
excluded_months = monthly.loc[COMPLETE_CUTOFF:].index.strftime("%Y-%m").tolist()[1:]

series = monthly_complete["quantity"]
series.index.freq = "MS"

# Time-aware split: last 6 complete months = test set
test_size = 6
train, test = series.iloc[:-test_size], series.iloc[-test_size:]

# Baseline: naive last-value-carried-forward + seasonal-naive alt.
naive_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
baseline_mae = mean_absolute_error(test, naive_pred)
baseline_rmse = np.sqrt(mean_squared_error(test, naive_pred))

# Model: Holt-Winters (additive trend, additive seasonality, period=12)
hw_model = ExponentialSmoothing(
    train, trend="add", seasonal="add", seasonal_periods=12,
    initialization_method="estimated",
).fit()
hw_pred = hw_model.forecast(test_size)
hw_mae = mean_absolute_error(test, hw_pred)
hw_rmse = np.sqrt(mean_squared_error(test, hw_pred))

# Refit on full complete series for a genuine forward forecast (next 6 months)
hw_full = ExponentialSmoothing(
    series, trend="add", seasonal="add", seasonal_periods=12,
    initialization_method="estimated",
).fit()
future_forecast = hw_full.forecast(6)

results = {
    "excluded_months_data_artifact": excluded_months,
    "train_months": len(train),
    "test_months": len(test),
    "baseline_naive_mae": round(float(baseline_mae), 1),
    "baseline_naive_rmse": round(float(baseline_rmse), 1),
    "holt_winters_mae": round(float(hw_mae), 1),
    "holt_winters_rmse": round(float(hw_rmse), 1),
    "improvement_over_baseline_pct": round(100 * (1 - hw_mae / baseline_mae), 1),
    "selected_model": "Holt-Winters (additive trend + seasonality)",
    "next_6_month_forecast": {str(k.date()): round(float(v), 1) for k, v in future_forecast.items()},
}

with open("reports/forecast_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot actual vs forecast
fig, ax = plt.subplots(figsize=(11, 5))
train.plot(ax=ax, label="Train (actual)", color="#2c6fbb")
test.plot(ax=ax, label="Test (actual)", color="#3f8f5f", marker="o")
hw_pred.plot(ax=ax, label="Holt-Winters forecast (test)", color="#c0392b", linestyle="--", marker="x")
future_forecast.plot(ax=ax, label="Forward forecast (next 6mo)", color="#b3763b", linestyle=":", marker="s")
ax.axvline(train.index[-1], color="gray", linestyle=":", alpha=0.6)
ax.set_title("Monthly Demand (Units) - Actual vs Holt-Winters Forecast")
ax.set_ylabel("Units Sold"); ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/15_forecast_actual_vs_predicted.png", bbox_inches="tight")
plt.close(fig)

print(json.dumps(results, indent=2))
