"""
Phase C - Supply Chain KPI Engineering.

Computes the KPI set required by Section 11 of the project guidelines at the
correct grain (order-level for order/revenue KPIs; line-item level where the
guideline explicitly calls for it, e.g. late-delivery rate).
"""
import pandas as pd
import json

df = pd.read_parquet("data/processed/dataco_clean.parquet")

# Order-level view: de-duplicate to one row per Order Id for order-level KPIs
order_level = df.drop_duplicates(subset="Order Id")

kpis = {}

kpis["total_revenue"] = round(df["Sales"].sum(), 2)
kpis["total_orders"] = int(df["Order Id"].nunique())
kpis["total_units_sold"] = int(df["Order Item Quantity"].sum())
kpis["average_order_value"] = round(df["Sales"].sum() / df["Order Id"].nunique(), 2)
kpis["average_shipping_cost_per_line_item"] = round(df["Order Item Discount"].mean(), 2)  # discount as cost proxy note below
kpis["average_delivery_time_days"] = round(df["Days for shipping (real)"].mean(), 2)

kpis["on_time_delivery_rate_pct"] = round(100 * (df["Late_delivery_risk"] == 0).mean(), 2)
kpis["late_delivery_rate_pct"] = round(100 * (df["Late_delivery_risk"] == 1).mean(), 2)

order_status_counts = df["Order Status"].value_counts()
cancel_rate = 100 * order_status_counts.get("CANCELED", 0) / len(df)
kpis["cancellation_rate_pct"] = round(cancel_rate, 2)

# Sales growth rate: first vs last complete month in range (excl. truncated tail)
monthly = df.groupby("order_year_month")["Sales"].sum().sort_index()
monthly_complete = monthly.iloc[:-4]  # drop last partial/truncated months (see forecasting phase)
growth_rate = 100 * (monthly_complete.iloc[-1] - monthly_complete.iloc[0]) / monthly_complete.iloc[0]
kpis["sales_growth_rate_pct_first_vs_last_complete_month"] = round(growth_rate, 2)

# Product/category contribution - top category share
cat_sales = df.groupby("Category Name")["Sales"].sum().sort_values(ascending=False)
kpis["top_category"] = cat_sales.index[0]
kpis["top_category_revenue_share_pct"] = round(100 * cat_sales.iloc[0] / cat_sales.sum(), 2)

# Regional performance - top region share
region_sales = df.groupby("Order Region")["Sales"].sum().sort_values(ascending=False)
kpis["top_region"] = region_sales.index[0]
kpis["top_region_revenue_share_pct"] = round(100 * region_sales.iloc[0] / region_sales.sum(), 2)

# Supplier performance proxy (Department Name) - see documented limitation
dept_perf = df.groupby("Department Name").agg(
    revenue=("Sales", "sum"),
    late_rate_pct=("Late_delivery_risk", lambda s: round(100 * s.mean(), 2)),
    n_line_items=("Order Item Id", "count"),
).sort_values("revenue", ascending=False)
kpis["top_department_proxy_supplier"] = dept_perf.index[0]

kpis["note_on_avg_shipping_cost"] = (
    "Dataset has no direct 'shipping cost' field. Order Item Discount is NOT "
    "a shipping cost - true shipping cost is not computable from this dataset "
    "and is documented as a limitation rather than approximated silently."
)
del kpis["average_shipping_cost_per_line_item"]

with open("reports/kpi_summary.json", "w") as f:
    json.dump(kpis, f, indent=2, default=str)

dept_perf.to_csv("reports/department_supplier_scorecard.csv")

print(json.dumps(kpis, indent=2, default=str))
