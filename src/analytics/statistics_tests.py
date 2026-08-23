"""
Phase G - Statistics: descriptives, correlation, outliers, hypothesis tests, CIs.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_parquet("data/processed/dataco_clean.parquet")
results = {}

# ---------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------
desc = df[["Sales", "Order Item Quantity", "Order Item Discount",
           "Days for shipping (real)"]].describe().round(2)
desc.to_csv("reports/descriptive_statistics.csv")
results["descriptive_statistics"] = desc.to_dict()

# ---------------------------------------------------------------------------
# 2. Correlation analysis
# ---------------------------------------------------------------------------
num_cols = ["Days for shipping (real)", "Days for shipment (scheduled)",
            "Sales", "Order Item Quantity", "Order Item Discount",
            "Order Item Profit Ratio", "Late_delivery_risk"]
corr = df[num_cols].corr().round(3)
corr.to_csv("reports/correlation_matrix.csv")

# ---------------------------------------------------------------------------
# 3. Outlier detection (IQR) - already flagged in Phase A; summarize rates
# ---------------------------------------------------------------------------
outlier_cols = [c for c in df.columns if c.startswith("outlier_")]
results["outlier_rate_pct"] = {c: round(100 * df[c].mean(), 2) for c in outlier_cols}

# ---------------------------------------------------------------------------
# 4. Hypothesis test A: Does shipping mode affect delivery time? (ANOVA)
# ---------------------------------------------------------------------------
groups = [g["Days for shipping (real)"].values for _, g in df.groupby("Shipping Mode")]
f_stat, p_val = stats.f_oneway(*groups)
results["anova_shipping_mode_vs_delivery_time"] = {
    "test": "One-way ANOVA",
    "H0": "Mean actual delivery days is equal across all shipping modes",
    "f_statistic": round(float(f_stat), 3),
    "p_value": float(p_val),
    "conclusion": (
        "Reject H0 (p < 0.001): shipping mode has a statistically significant "
        "relationship with delivery time." if p_val < 0.05 else
        "Fail to reject H0: no significant relationship found."
    ),
}

# ---------------------------------------------------------------------------
# 5. Hypothesis test B: Does customer segment predict late delivery? (Chi-sq)
# ---------------------------------------------------------------------------
contingency = pd.crosstab(df["Customer Segment"], df["Late_delivery_risk"])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
results["chisquare_segment_vs_late_delivery"] = {
    "test": "Chi-square test of independence",
    "H0": "Customer segment and late-delivery risk are independent",
    "chi2_statistic": round(float(chi2), 3),
    "dof": int(dof),
    "p_value": float(p_chi),
    "conclusion": (
        "Fail to reject H0: customer segment does not predict late delivery."
        if p_chi >= 0.05 else
        "Reject H0: customer segment is significantly associated with late delivery."
    ),
}

# ---------------------------------------------------------------------------
# 6. Confidence interval: mean delivery delay (95% CI)
# ---------------------------------------------------------------------------
delay = df["shipping_delay_days"].dropna()
mean_delay = delay.mean()
sem = stats.sem(delay)
ci = stats.t.interval(0.95, len(delay) - 1, loc=mean_delay, scale=sem)
results["confidence_interval_mean_shipping_delay_days"] = {
    "mean": round(float(mean_delay), 4),
    "95pct_ci_lower": round(float(ci[0]), 4),
    "95pct_ci_upper": round(float(ci[1]), 4),
}

with open("reports/statistical_test_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps({k: v for k, v in results.items() if k != "descriptive_statistics"}, indent=2, default=str))
