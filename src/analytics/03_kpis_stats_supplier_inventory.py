"""
Phase C - Supply Chain KPIs
Phase E - Supplier Performance Analysis (Department Name used as documented proxy)
Phase F - Inventory Analytics (demand-based, since no stock-level field exists)
Phase G - Statistics (descriptive stats, correlation, outliers, hypothesis test)
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_csv("data/processed/cleaned_supply_chain.csv", parse_dates=["order date (DateOrders)"])
out = {}

# =====================================================================
# PHASE C - SUPPLY CHAIN KPIs
# =====================================================================
n_orders = df["Order Id"].nunique()
kpis = {
    "Total Revenue ($)": round(df["Sales"].sum(), 2),
    "Total Orders": int(n_orders),
    "Total Units Sold": int(df["Order Item Quantity"].sum()),
    "Average Order Value ($)": round(df["Sales"].sum() / n_orders, 2),
    "Average Discount per Item ($)": round(df["Order Item Discount"].mean(), 2),
    "Average Delivery Time (days)": round(df["Days for shipping (real)"].mean(), 2),
    "On-Time Delivery Rate (%)": round(100 * df["Delivery Status"].isin(
        ["Shipping on time", "Advance shipping"]).mean(), 2),
    "Late Delivery Rate (%)": round(100 * df["is_late"].mean(), 2),
    "Cancellation Rate (%)": round(100 * df[df["Order Status"] == "CANCELED"]["Order Id"].nunique() / n_orders, 2),
    "Sales Growth Rate (last vs first full month, %)": None,  # filled below
}
monthly = df.groupby("order_yyyymm")["Sales"].sum().sort_index()
if len(monthly) > 1:
    kpis["Sales Growth Rate (last vs first full month, %)"] = round(
        100 * (monthly.iloc[-2] - monthly.iloc[0]) / monthly.iloc[0], 2)  # -2 to skip partial last month

top_cat = df.groupby("Category Name")["Sales"].sum().idxmax()
top_region = df.groupby("Order Region")["Sales"].sum().idxmax()
kpis["Top Category by Sales"] = top_cat
kpis["Top Region by Sales"] = top_region
out["phase_C_kpis"] = kpis

# =====================================================================
# PHASE E - SUPPLIER PERFORMANCE ANALYSIS
# LIMITATION: The DataCo dataset has no dedicated Supplier entity/table.
# 'Department Name' is used here as the closest available grouping to
# approximate a supplying business unit. This substitution is explicit
# and should not be presented to stakeholders as literal supplier data.
# =====================================================================
dept = df.groupby("Department Name").agg(
    order_volume=("Order Item Id", "count"),
    sales_contribution=("Sales", "sum"),
    avg_fulfillment_days=("Days for shipping (real)", "mean"),
    late_delivery_pct=("is_late", lambda x: round(100 * x.mean(), 2)),
).reset_index()
dept["sales_contribution_pct"] = round(100 * dept["sales_contribution"] / dept["sales_contribution"].sum(), 2)
dept["avg_fulfillment_days"] = dept["avg_fulfillment_days"].round(2)
dept = dept.sort_values("sales_contribution", ascending=False)
dept["performance_rank"] = range(1, len(dept) + 1)
dept["risk_category"] = pd.cut(dept["late_delivery_pct"], bins=[-1, 45, 55, 100],
                                labels=["Lower Risk", "Moderate Risk", "Higher Risk"])
out["phase_E_department_scorecard_NOTE"] = (
    "No Supplier field exists in source data; Department Name used as documented proxy. "
    "Defect/return indicators and independent cost data are NOT available and are not reported."
)
out["phase_E_department_scorecard"] = dept.round(2).to_dict(orient="records")

# =====================================================================
# PHASE F - INVENTORY ANALYTICS (demand-based only; no stock-level field)
# =====================================================================
prod = df.groupby("Product Name").agg(
    total_qty_sold=("Order Item Quantity", "sum"),
    total_sales=("Sales", "sum"),
    n_orders=("Order Id", "nunique"),
).reset_index()
# Demand variability: coefficient of variation of monthly quantity per product
monthly_qty = df.groupby(["Product Name", "order_yyyymm"])["Order Item Quantity"].sum().reset_index()
cv = monthly_qty.groupby("Product Name")["Order Item Quantity"].agg(["mean", "std"]).reset_index()
cv["demand_cv"] = (cv["std"] / cv["mean"]).round(2)
prod = prod.merge(cv[["Product Name", "demand_cv"]], on="Product Name", how="left")
prod["demand_cv"] = prod["demand_cv"].fillna(0)
prod = prod.sort_values("total_qty_sold", ascending=False)
prod["mover_class"] = pd.qcut(prod["total_qty_sold"], q=[0, .3, .7, 1.0],
                               labels=["Slow Mover", "Medium Mover", "Fast Mover"])
prod["stockout_risk_flag"] = np.where(
    (prod["mover_class"] == "Fast Mover") & (prod["demand_cv"] > prod["demand_cv"].median()),
    "High (fast-moving + volatile demand)", "Lower"
)
out["phase_F_inventory_NOTE"] = (
    "No on-hand stock quantity field exists in source data. Metrics below are DEMAND-BASED "
    "(historical sales/quantity), not literal inventory/stock-level calculations."
)
out["phase_F_top20_fast_movers"] = prod.head(20)[
    ["Product Name", "total_qty_sold", "total_sales", "demand_cv", "mover_class", "stockout_risk_flag"]
].round(2).to_dict(orient="records")
out["phase_F_bottom10_slow_movers"] = prod.tail(10)[
    ["Product Name", "total_qty_sold", "total_sales", "demand_cv", "mover_class"]
].round(2).to_dict(orient="records")

# =====================================================================
# PHASE G - STATISTICS
# =====================================================================
desc_cols = ["Sales", "Order Item Quantity", "Order Item Discount", "Days for shipping (real)"]
out["phase_G_descriptive_stats"] = df[desc_cols].describe().round(2).to_dict()

corr_cols = ["Days for shipping (real)", "Days for shipment (scheduled)", "Sales",
             "Order Item Quantity", "Order Item Discount", "Benefit per order"]
out["phase_G_correlation_matrix"] = df[corr_cols].corr().round(3).to_dict()

# Outlier detection (IQR) - Sales
q1, q3 = df["Sales"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out["phase_G_sales_outliers_iqr"] = {
    "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
    "n_outliers": int(((df["Sales"] < lower) | (df["Sales"] > upper)).sum()),
    "pct_outliers": round(100 * ((df["Sales"] < lower) | (df["Sales"] > upper)).mean(), 2),
}

# Hypothesis test: does shipping mode affect delivery time? (one-way ANOVA)
groups = [g["Days for shipping (real)"].values for _, g in df.groupby("Shipping Mode")]
f_stat, p_val = stats.f_oneway(*groups)
out["phase_G_hypothesis_test_shipping_mode_vs_delivery_time"] = {
    "test": "One-way ANOVA",
    "H0": "Mean delivery time is equal across all shipping modes",
    "F_statistic": round(f_stat, 4),
    "p_value": p_val,
    "conclusion": "Reject H0 - shipping mode has a statistically significant relationship "
                  "with delivery time (p < 0.05)" if p_val < 0.05 else
                  "Fail to reject H0 - no statistically significant relationship found",
    "alpha": 0.05,
}

# 95% CI for mean delivery time
mean_days = df["Days for shipping (real)"].mean()
sem = stats.sem(df["Days for shipping (real)"])
ci = stats.t.interval(0.95, len(df) - 1, loc=mean_days, scale=sem)
out["phase_G_ci_avg_delivery_time"] = {
    "mean_days": round(mean_days, 3),
    "95pct_ci_lower": round(ci[0], 3),
    "95pct_ci_upper": round(ci[1], 3),
}

# Hypothesis test 2: is late delivery associated with customer segment? (chi-square)
ct = pd.crosstab(df["Customer Segment"], df["is_late"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
out["phase_G_hypothesis_test_segment_vs_late"] = {
    "test": "Chi-square test of independence",
    "H0": "Late delivery is independent of customer segment",
    "chi2_statistic": round(chi2, 4),
    "p_value": p_chi,
    "conclusion": "Reject H0 - late delivery rate differs significantly by customer segment" if p_chi < 0.05
                  else "Fail to reject H0",
}

with open("/home/claude/project/reports/phaseC_E_F_G_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("KPIs:")
for k, v in kpis.items():
    print(f"  {k}: {v}")
print("\nHypothesis test (shipping mode vs delivery time): p =", p_val)
print("Hypothesis test (segment vs late delivery): p =", p_chi)
print("\nSaved full results to reports/phaseC_E_F_G_results.json")
