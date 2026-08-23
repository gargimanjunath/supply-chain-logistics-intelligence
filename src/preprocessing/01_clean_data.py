"""
Phase A - Data Cleaning & Preparation
Supply Chain & Logistics Intelligence System

Loads the raw DataCo Smart Supply Chain CSV, documents the data grain,
handles missing values, standardizes types, flags outliers, and writes
a cleaned analytical dataset. Raw data is left untouched.
"""
import pandas as pd
import numpy as np
import json
import os

RAW_PATH = "data/raw/DataCoSupplyChainDataset.csv"
OUT_DIR = "data/processed"
LOG_PATH = "reports/phase_a_cleaning_log.json"
os.makedirs(OUT_DIR, exist_ok=True)

log = {}

# ---------------------------------------------------------------
# 1. Load raw data (ISO-8859-1 / latin1 encoded)
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH, encoding="latin1")
log["raw_shape"] = list(df.shape)

# ---------------------------------------------------------------
# 2. Data grain
# One row = one ORDER ITEM (line item within an order), NOT one order.
# 'Order Item Id' is the unique key. 'Order Id' repeats across items
# that belong to the same order (avg ~2.75 items/order in this dataset).
# All order-level KPIs (e.g. Total Orders) must use nunique('Order Id'),
# not len(df).
# ---------------------------------------------------------------
assert df["Order Item Id"].is_unique, "Grain assumption violated: Order Item Id not unique"
log["grain"] = "one row = one order line item; unique key = Order Item Id"
log["unique_orders"] = int(df["Order Id"].nunique())
log["avg_items_per_order"] = round(len(df) / df["Order Id"].nunique(), 3)

# ---------------------------------------------------------------
# 3. Drop columns that are unusable or a privacy risk
#    - Product Description: 100% null in this file
#    - Customer Email / Password / Fname / Lname / Street: PII, not
#      needed for supply-chain analytics
#    - Order Zipcode: ~86% null, not usable
#    - Product Image: URL, no analytical value
# ---------------------------------------------------------------
drop_cols = [
    "Product Description", "Customer Email", "Customer Password",
    "Customer Fname", "Customer Lname", "Customer Street",
    "Order Zipcode", "Product Image",
]
dropped = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=dropped)
log["dropped_columns"] = dropped

# ---------------------------------------------------------------
# 4. Missing values remaining (Customer Lname already dropped above;
#    Customer Zipcode has 3 nulls - fill with 0/unknown marker)
# ---------------------------------------------------------------
missing_before = df.isna().sum()
missing_before = missing_before[missing_before > 0].to_dict()
log["missing_values_before_fill"] = {k: int(v) for k, v in missing_before.items()}

if "Customer Zipcode" in df.columns:
    df["Customer Zipcode"] = df["Customer Zipcode"].fillna(0).astype(int)

# ---------------------------------------------------------------
# 5. Duplicates - distinguish true duplicate rows from repeated
#    order items (which are expected and valid)
# ---------------------------------------------------------------
full_dupes = df.duplicated().sum()
log["fully_duplicated_rows"] = int(full_dupes)
df = df.drop_duplicates()

# ---------------------------------------------------------------
# 6. Convert dates to datetime
# ---------------------------------------------------------------
df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
log["bad_order_dates"] = int(df["order date (DateOrders)"].isna().sum())
log["bad_shipping_dates"] = int(df["shipping date (DateOrders)"].isna().sum())

# ---------------------------------------------------------------
# 7. Standardize categorical text fields
# ---------------------------------------------------------------
cat_cols = ["Type", "Delivery Status", "Category Name", "Customer Segment",
            "Department Name", "Market", "Order Status", "Order Region",
            "Shipping Mode", "Order Country", "Customer Country"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# ---------------------------------------------------------------
# 8. Invalid / suspicious numeric values
#    - Negative sales, prices, or quantities are not physically valid
#    - Days for shipping (real) should be >= 0
# ---------------------------------------------------------------
numeric_checks = {
    "Sales": (df["Sales"] < 0).sum(),
    "Order Item Quantity": (df["Order Item Quantity"] <= 0).sum(),
    "Product Price": (df["Product Price"] < 0).sum(),
    "Days for shipping (real)": (df["Days for shipping (real)"] < 0).sum(),
}
log["suspicious_numeric_rows"] = {k: int(v) for k, v in numeric_checks.items()}
# None found to be negative in this dataset in practice, but we guard
# against them for reproducibility on any future data refresh.
df = df[df["Sales"] >= 0]
df = df[df["Order Item Quantity"] > 0]

# ---------------------------------------------------------------
# 9. Outlier flags (IQR method) on key numeric fields - flagged, not
#    removed, so downstream analysis can choose to exclude them.
# ---------------------------------------------------------------
def iqr_flag(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

outlier_summary = {}
for col in ["Sales", "Order Item Quantity", "Days for shipping (real)", "Benefit per order"]:
    flag = iqr_flag(df[col])
    df[f"outlier_{col.replace(' ', '_')}"] = flag
    outlier_summary[col] = int(flag.sum())
log["iqr_outlier_counts"] = outlier_summary

# ---------------------------------------------------------------
# 10. Derived fields used throughout the project
# ---------------------------------------------------------------
df["order_year"] = df["order date (DateOrders)"].dt.year
df["order_month"] = df["order date (DateOrders)"].dt.to_period("M").astype(str)
df["order_week"] = df["order date (DateOrders)"].dt.to_period("W").astype(str)
df["order_yyyymm"] = df["order date (DateOrders)"].dt.strftime("%Y-%m")
df["is_late"] = (df["Delivery Status"] == "Late delivery").astype(int)
df["is_cancelled"] = df["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"]).astype(int)
df["shipping_delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

# ---------------------------------------------------------------
# 11. Cancelled / returned / incomplete orders - documented, not removed
#     (needed for cancellation-rate KPI and status-aware analysis)
# ---------------------------------------------------------------
log["order_status_breakdown"] = df["Order Status"].value_counts().to_dict()
log["delivery_status_breakdown"] = df["Delivery Status"].value_counts().to_dict()

# ---------------------------------------------------------------
# 12. Known dataset limitations (documented per project guidelines -
#     do not invent metrics the data cannot support)
# ---------------------------------------------------------------
log["limitations"] = [
    "No explicit 'Supplier' entity/table exists in this dataset. "
    "'Department Name' is the closest available grouping and is used "
    "as a proxy for supplier/business-unit performance in Phase E, "
    "with this substitution explicitly documented wherever it is used.",
    "No physical on-hand inventory / stock-level field exists. Phase F "
    "(Inventory Analytics) is therefore demand-based (sales & quantity "
    "history) rather than true stock-level inventory analysis.",
    "No product cost / COGS field exists beyond 'Benefit per order' and "
    "'Order Item Profit Ratio', so supplier cost-competitiveness cannot "
    "be independently verified.",
]

# ---------------------------------------------------------------
# 13. Save cleaned dataset (processed layer, raw untouched)
# ---------------------------------------------------------------
out_csv = os.path.join(OUT_DIR, "cleaned_supply_chain.csv")
df.to_csv(out_csv, index=False)
log["cleaned_shape"] = list(df.shape)
log["output_file"] = out_csv

with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2, default=str)

print("Cleaning complete.")
print(json.dumps(log, indent=2, default=str))
