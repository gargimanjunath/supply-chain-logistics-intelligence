"""
Phase F - Inventory Analytics (demand-based proxies).

The dataset has NO on-hand stock/inventory quantity field. This script
therefore builds DEMAND-BASED indicators only, explicitly labelled as such:
- Sales velocity (units/month) -> fast vs. slow movers
- Demand variability (coefficient of variation across months) -> stock-out risk proxy
- Combined replenishment-priority ranking
"""
import pandas as pd
import json

df = pd.read_parquet("data/processed/dataco_clean.parquet")

# Restrict to complete months only (exclude truncation artifact tail)
complete = df[df["order_year_month"] <= "2017-09"]

monthly_product = complete.groupby(["Product Name", "order_year_month"])["Order Item Quantity"].sum().unstack(fill_value=0)
n_months = monthly_product.shape[1]

velocity = monthly_product.sum(axis=1) / n_months
variability = monthly_product.std(axis=1) / monthly_product.mean(axis=1).replace(0, pd.NA)

demand_profile = pd.DataFrame({
    "avg_units_per_month": velocity.round(2),
    "demand_coefficient_of_variation": variability.round(3),
    "total_units_sold": monthly_product.sum(axis=1),
}).dropna()

demand_profile["mover_class"] = pd.qcut(
    demand_profile["avg_units_per_month"], q=[0, 0.33, 0.66, 1.0],
    labels=["Slow Mover", "Medium Mover", "Fast Mover"]
)
demand_profile["stockout_risk_proxy"] = pd.qcut(
    demand_profile["demand_coefficient_of_variation"], q=[0, 0.66, 1.0],
    labels=["Lower Variability", "High Variability (Higher Risk Proxy)"]
)

demand_profile = demand_profile.sort_values("avg_units_per_month", ascending=False)
demand_profile.to_csv("reports/inventory_demand_profile.csv")

summary = {
    "note": (
        "No on-hand inventory/stock field exists in this dataset. These are "
        "DEMAND-BASED indicators (order/sales velocity and variability), not "
        "measured stock levels, and are presented as such."
    ),
    "n_products_profiled": len(demand_profile),
    "fast_movers_count": int((demand_profile["mover_class"] == "Fast Mover").sum()),
    "slow_movers_count": int((demand_profile["mover_class"] == "Slow Mover").sum()),
    "top_5_fast_movers": demand_profile.head(5)["avg_units_per_month"].to_dict(),
}
with open("reports/inventory_analytics_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
