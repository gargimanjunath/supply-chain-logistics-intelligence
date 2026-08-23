"""
Phase A - Data Cleaning & Preparation
Supply Chain & Logistics Intelligence System

Reads the raw DataCo Smart Supply Chain CSV, documents data-quality findings,
cleans/standardizes it, and writes a processed parquet + CSV for downstream phases.
Raw data is never modified in place (kept separate from processed data).
"""
import pandas as pd
import numpy as np
import json
import os

RAW_PATH = "data/raw/DataCoSupplyChainDataset.csv"
OUT_CSV = "data/processed/dataco_clean.csv"
OUT_PARQUET = "data/processed/dataco_clean.parquet"
LOG_PATH = "logs/phase_a_cleaning_log.json"

log = {}

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH, encoding="latin1")
log["raw_shape"] = list(df.shape)

# ---------------------------------------------------------------------------
# 2. Grain check (Section 6 of guidelines)
# One row = one order LINE ITEM, not one order. Confirm via Order Id repeats.
# ---------------------------------------------------------------------------
n_rows = len(df)
n_unique_orders = df["Order Id"].nunique()
n_unique_order_items = df["Order Item Id"].nunique()
log["grain_check"] = {
    "n_rows": int(n_rows),
    "n_unique_order_ids": int(n_unique_orders),
    "n_unique_order_item_ids": int(n_unique_order_items),
    "conclusion": (
        "One row = one order line item (Order Item Id is unique per row; "
        "Order Id repeats across multiple line items). Counting rows as "
        "'orders' would overstate order volume by roughly "
        f"{round((n_rows/n_unique_orders - 1)*100,1)}%."
    ),
}

# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
log["missing_values_before"] = {k: int(v) for k, v in missing.items()}

# Columns that are almost entirely empty / not usable for analysis (PII, or
# structurally near-empty) are dropped. "Order Zipcode" is >80% null and not
# needed since Order City/State/Country/Region already give geography.
# "Product Description" and "Product Image" are unused free-text/URL fields.
cols_to_drop = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Product Description", "Product Image", "Order Zipcode",
]
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)
log["columns_dropped_pii_or_unusable"] = cols_to_drop

# Customer Lname/Zipcode etc had few nulls; Customer Zipcode kept but any
# residual nulls imputed with -1 flag (not used analytically, geo covered by
# Customer City/State/Country).
if "Customer Zipcode" in df.columns:
    df["Customer Zipcode"] = df["Customer Zipcode"].fillna(-1)

# ---------------------------------------------------------------------------
# 4. Duplicates vs. repeated order items (Section 8)
# ---------------------------------------------------------------------------
exact_dupe_rows = df.duplicated().sum()
log["exact_duplicate_rows"] = int(exact_dupe_rows)
if exact_dupe_rows > 0:
    df = df.drop_duplicates()

# ---------------------------------------------------------------------------
# 5. Date conversion
# ---------------------------------------------------------------------------
df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["shipping date (DateOrders)"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")

bad_dates = df["order date (DateOrders)"].isna().sum() + df["shipping date (DateOrders)"].isna().sum()
log["unparseable_dates"] = int(bad_dates)

# ---------------------------------------------------------------------------
# 6. Invalid / suspicious numeric values
# ---------------------------------------------------------------------------
neg_sales = int((df["Sales"] < 0).sum())
neg_qty = int((df["Order Item Quantity"] < 0).sum())
neg_price = int((df["Product Price"] < 0).sum())
log["invalid_numeric_flags"] = {
    "negative_sales_rows": neg_sales,
    "negative_quantity_rows": neg_qty,
    "negative_price_rows": neg_price,
}
# DataCo Sales/Quantity/Price are all >=0 in this dataset; no rows removed
# here, values are just flagged in the log for transparency.

# Shipping duration sanity: real days should be >=0
df["Days for shipping (real)"] = df["Days for shipping (real)"].clip(lower=0)

# ---------------------------------------------------------------------------
# 7. Standardize categoricals (trim whitespace, fix casing)
# ---------------------------------------------------------------------------
cat_cols = [
    "Type", "Delivery Status", "Category Name", "Customer Segment", "Market",
    "Order Status", "Order Region", "Shipping Mode", "Department Name",
    "Customer Country", "Customer City", "Customer State", "Order Country",
    "Order City", "Order State",
]
for c in cat_cols:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].str.strip()

# ---------------------------------------------------------------------------
# 8. Cancelled / returned / incomplete orders (Section 8)
# ---------------------------------------------------------------------------
status_counts = df["Order Status"].value_counts().to_dict()
log["order_status_distribution"] = {k: int(v) for k, v in status_counts.items()}
cancel_like = df["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"])
log["cancelled_or_fraud_rows"] = int(cancel_like.sum())
# These rows are RETAINED in the cleaned dataset (they are real business
# events) but flagged with a boolean column so downstream analysis (e.g.
# revenue KPIs, ML training) can choose to include/exclude them explicitly.
df["is_cancelled_or_fraud"] = cancel_like

# ---------------------------------------------------------------------------
# 9. Outlier detection (IQR) on key numeric fields - flag, do not silently drop
# ---------------------------------------------------------------------------
def iqr_flags(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

outlier_summary = {}
for col in ["Sales", "Order Item Quantity", "Order Item Discount", "Days for shipping (real)"]:
    flag_col = f"outlier_{col.replace(' ', '_')}"
    df[flag_col] = iqr_flags(df[col])
    outlier_summary[col] = int(df[flag_col].sum())
log["outlier_counts_iqr"] = outlier_summary
# Outliers are flagged, not deleted -- legitimate large B2B orders exist.

# ---------------------------------------------------------------------------
# 10. Feature engineering used across later phases
# ---------------------------------------------------------------------------
df["order_year"] = df["order date (DateOrders)"].dt.year
df["order_month"] = df["order date (DateOrders)"].dt.month
df["order_year_month"] = df["order date (DateOrders)"].dt.to_period("M").astype(str)
df["order_dow"] = df["order date (DateOrders)"].dt.day_name()
df["shipping_delay_days"] = (
    df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
)

# ---------------------------------------------------------------------------
# 11. Documented structural limitations (Section 13/14 of guidelines)
# ---------------------------------------------------------------------------
log["documented_limitations"] = {
    "supplier_entity": (
        "The dataset has no dedicated Supplier table/ID. 'Department Name' "
        "(the selling department/store) is used as a supplier-performance "
        "proxy throughout Phase E. This is explicitly disclosed in the "
        "report and dashboard, not presented as true supplier data."
    ),
    "inventory_stock_levels": (
        "The dataset has no on-hand inventory/stock quantity field. Phase F "
        "therefore builds DEMAND-BASED indicators (order frequency, sales "
        "velocity, demand variability) as a proxy for fast/slow movers and "
        "stock-out risk, rather than claiming to measure real inventory."
    ),
}

os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs", exist_ok=True)
df.to_csv(OUT_CSV, index=False)
df.to_parquet(OUT_PARQUET, index=False)

log["processed_shape"] = list(df.shape)
log["processed_columns"] = df.columns.tolist()

with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2, default=str)

print("Cleaning complete.")
print("Raw shape:", log["raw_shape"], "-> Processed shape:", log["processed_shape"])
print("Log written to", LOG_PATH)
