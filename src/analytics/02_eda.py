"""Phase D - EDA & Business Analysis. Generates charts to reports/figures/."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("data/processed/cleaned_supply_chain.csv", parse_dates=["order date (DateOrders)"])

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=130)
    plt.close(fig)

# 1. Monthly sales trend
monthly = df.groupby("order_yyyymm")["Sales"].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(monthly["order_yyyymm"], monthly["Sales"], marker="o", color="#2E5EAA")
ax.set_title("Monthly Sales Trend")
ax.set_xlabel("Month"); ax.set_ylabel("Sales ($)")
ax.set_xticks(ax.get_xticks()[::3])
save(fig, "01_monthly_sales_trend")

# 2. Category performance
cat = df.groupby("Category Name")["Sales"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=cat.values, y=cat.index, ax=ax, color="#2E5EAA")
ax.set_title("Top 10 Categories by Sales")
ax.set_xlabel("Sales ($)")
save(fig, "02_top_categories")

# 3. Regional sales distribution
reg = df.groupby("Order Region")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=reg.values, y=reg.index, ax=ax, color="#4C8C4A")
ax.set_title("Sales by Order Region")
ax.set_xlabel("Sales ($)")
save(fig, "03_regional_sales")

# 4. Shipping mode comparison (count + avg delivery days)
mode_stats = df.groupby("Shipping Mode").agg(count=("Order Item Id", "count"),
                                              avg_days=("Days for shipping (real)", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=mode_stats, x="Shipping Mode", y="count", ax=axes[0], color="#E07B39")
axes[0].set_title("Line Items by Shipping Mode"); axes[0].tick_params(axis='x', rotation=20)
sns.barplot(data=mode_stats, x="Shipping Mode", y="avg_days", ax=axes[1], color="#C1443C")
axes[1].set_title("Avg Delivery Days by Shipping Mode"); axes[1].tick_params(axis='x', rotation=20)
save(fig, "04_shipping_mode_comparison")

# 5. Shipping cost / discount distribution (using Order Item Discount as cost proxy)
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["Order Item Discount"], bins=40, ax=ax, color="#7A5FA0")
ax.set_title("Order Item Discount Distribution")
save(fig, "05_discount_distribution")

# 6. Delivery time distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["Days for shipping (real)"], bins=15, ax=ax, color="#2E5EAA")
ax.set_title("Delivery Time Distribution (Days)")
save(fig, "06_delivery_time_distribution")

# 7. Late delivery rate by region (top 10 by volume)
top_regions = df["Order Region"].value_counts().head(10).index
late_by_region = df[df["Order Region"].isin(top_regions)].groupby("Order Region")["is_late"].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=late_by_region.values, y=late_by_region.index, ax=ax, color="#C1443C")
ax.set_title("Late Delivery Rate by Region (%) - Top 10 Regions by Volume")
ax.set_xlabel("Late Delivery Rate (%)")
save(fig, "07_late_delivery_by_region")

# 8. Customer segment performance
seg = df.groupby("Customer Segment").agg(sales=("Sales", "sum"), orders=("Order Id", "nunique")).reset_index()
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=seg, x="Customer Segment", y="sales", ax=ax, color="#4C8C4A")
ax.set_title("Sales by Customer Segment")
save(fig, "08_customer_segment_sales")

# 9. Correlation heatmap of operational variables
num_cols = ["Days for shipping (real)", "Days for shipment (scheduled)", "Sales",
            "Order Item Quantity", "Order Item Discount", "Benefit per order",
            "Order Item Profit Ratio", "shipping_delay_days"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
ax.set_title("Correlation Matrix - Operational Variables")
save(fig, "09_correlation_heatmap")

# 10. Outlier boxplot - Sales
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x=df["Sales"], ax=ax, color="#E07B39")
ax.set_title("Sales Distribution & Outliers (IQR method)")
save(fig, "10_sales_outliers_boxplot")

print("EDA complete. Charts saved to", FIG_DIR)
print(os.listdir(FIG_DIR))
