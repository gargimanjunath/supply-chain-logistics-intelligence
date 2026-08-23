"""
Phase D - EDA & Business Analysis charts.
Each chart is built to answer a specific business question (Section 24 rule).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
FIG_DIR = "reports/figures"

df = pd.read_parquet("data/processed/dataco_clean.parquet")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 1. Monthly sales trend (order volume/trend question)
monthly = df.groupby("order_year_month")["Sales"].sum()
fig, ax = plt.subplots(figsize=(10, 4))
monthly.plot(ax=ax, marker="o", color="#2c6fbb")
ax.set_title("Monthly Sales Trend (Jan 2015 - Jan 2018)")
ax.set_ylabel("Total Sales ($)"); ax.set_xlabel("Month")
ax.tick_params(axis="x", rotation=75)
save(fig, "01_monthly_sales_trend")

# 2. Category performance
fig, ax = plt.subplots(figsize=(9, 6))
cat_sales = df.groupby("Category Name")["Sales"].sum().sort_values(ascending=False).head(12)
sns.barplot(x=cat_sales.values, y=cat_sales.index, ax=ax, color="#2c6fbb")
ax.set_title("Top 12 Categories by Revenue")
ax.set_xlabel("Total Sales ($)")
save(fig, "02_top_categories_by_revenue")

# 3. Regional sales distribution
fig, ax = plt.subplots(figsize=(9, 6))
region_sales = df.groupby("Order Region")["Sales"].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.values, y=region_sales.index, ax=ax, color="#3f8f5f")
ax.set_title("Revenue by Order Region")
ax.set_xlabel("Total Sales ($)")
save(fig, "03_revenue_by_region")

# 4. Shipping mode comparison (volume)
fig, ax = plt.subplots(figsize=(6, 4))
df["Shipping Mode"].value_counts().plot(kind="bar", ax=ax, color="#b3763b")
ax.set_title("Order Line Items by Shipping Mode")
ax.set_ylabel("Count"); ax.tick_params(axis="x", rotation=20)
save(fig, "04_shipping_mode_volume")

# 5. Shipping mode vs delivery time (relationship question - feeds ANOVA)
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="Shipping Mode", y="Days for shipping (real)", ax=ax)
ax.set_title("Actual Delivery Days by Shipping Mode")
save(fig, "05_delivery_days_by_shipping_mode")

# 6. Shipping-cost distribution -- dataset has no shipping cost field; use
#    Order Item Discount as the closest available cost-side distribution and
#    label the chart accordingly (documented, not silently substituted).
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["Order Item Discount"], bins=50, ax=ax, color="#8e5fb3")
ax.set_title("Order Item Discount Distribution\n(no direct shipping-cost field exists in this dataset)")
ax.set_xlabel("Order Item Discount ($)")
save(fig, "06_discount_distribution")

# 7. Delivery-time distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["Days for shipping (real)"], bins=20, ax=ax, color="#2c6fbb")
ax.set_title("Distribution of Actual Delivery Days")
ax.set_xlabel("Days for Shipping (real)")
save(fig, "07_delivery_time_distribution")

# 8. Late delivery rate by region (top 10)
late_by_region = df.groupby("Order Region")["Late_delivery_risk"].mean().sort_values(ascending=False).head(10) * 100
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=late_by_region.values, y=late_by_region.index, ax=ax, color="#c0392b")
ax.set_title("Top 10 Regions by Late-Delivery Rate")
ax.set_xlabel("Late-Delivery Rate (%)")
save(fig, "08_late_delivery_rate_by_region")

# 9. Customer segment performance
fig, ax = plt.subplots(figsize=(6, 4))
seg_sales = df.groupby("Customer Segment")["Sales"].sum().sort_values(ascending=False)
sns.barplot(x=seg_sales.index, y=seg_sales.values, ax=ax, color="#3f8f5f")
ax.set_title("Revenue by Customer Segment")
ax.set_ylabel("Total Sales ($)")
save(fig, "09_revenue_by_customer_segment")

# 10. Correlation heatmap among key operational variables
num_cols = ["Days for shipping (real)", "Days for shipment (scheduled)",
            "Sales", "Order Item Quantity", "Order Item Discount",
            "Order Item Profit Ratio", "Late_delivery_risk"]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Among Key Operational Variables")
save(fig, "10_correlation_heatmap")

# 11. Outlier analysis - Sales boxplot
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(y=df["Sales"], ax=ax, color="#b3763b")
ax.set_title("Sales Outlier Analysis (IQR-based)")
save(fig, "11_sales_outlier_boxplot")

# 12. Order status distribution
fig, ax = plt.subplots(figsize=(8, 4))
df["Order Status"].value_counts().plot(kind="bar", ax=ax, color="#2c6fbb")
ax.set_title("Order Status Distribution")
ax.tick_params(axis="x", rotation=45)
save(fig, "12_order_status_distribution")

# 13. Late-delivery rate trend over time (bottleneck identification)
late_trend = df.groupby("order_year_month")["Late_delivery_risk"].mean() * 100
fig, ax = plt.subplots(figsize=(10, 4))
late_trend.plot(ax=ax, marker="o", color="#c0392b")
ax.set_title("Late-Delivery Rate Trend Over Time")
ax.set_ylabel("Late-Delivery Rate (%)")
ax.tick_params(axis="x", rotation=75)
save(fig, "13_late_delivery_trend")

# 14. Discount rate vs profit ratio (operational relationship)
fig, ax = plt.subplots(figsize=(7, 5))
sample = df.sample(5000, random_state=42)
sns.scatterplot(data=sample, x="Order Item Discount Rate", y="Order Item Profit Ratio",
                 hue="Late_delivery_risk", alpha=0.4, ax=ax, palette=["#2c6fbb", "#c0392b"])
ax.set_title("Discount Rate vs Profit Ratio (sampled 5,000 line items)")
save(fig, "14_discount_vs_profit")

print("\nAll 14 EDA charts generated in", FIG_DIR)
