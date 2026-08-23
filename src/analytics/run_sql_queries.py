"""
Phase B - SQL Analysis execution.

The business queries are authored in MySQL syntax (sql/business_queries.sql,
per the project's recommended stack). SQLite does not support DATE_FORMAT,
so this runner uses SQLite-compatible equivalents (strftime) against a local
SQLite copy of the cleaned data and saves each query's result to CSV.
"""
import sqlite3
import pandas as pd
import os

DB_PATH = "data/processed/dataco.db"
OUT_DIR = "reports/sql_results"
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# Build normalized dimension tables so real JOINs can be demonstrated,
# rather than only querying the flat fact table.
conn.execute("DROP TABLE IF EXISTS dim_product")
conn.execute("""
    CREATE TABLE dim_product AS
    SELECT DISTINCT
        "Product Card Id",
        "Product Name",
        "Category Name",
        "Department Name",
        "Product Price"
    FROM orders_fact
""")
conn.execute("DROP TABLE IF EXISTS dim_customer")
conn.execute("""
    CREATE TABLE dim_customer AS
    SELECT DISTINCT
        "Order Customer Id" AS "Customer Id",
        "Customer Segment",
        "Customer Country",
        "Customer State",
        "Customer City"
    FROM orders_fact
""")
conn.commit()

queries = {
"Q1_totals": """
    SELECT
        ROUND(SUM(f.Sales), 2) AS total_sales,
        COUNT(DISTINCT f."Order Id") AS total_orders,
        SUM(f."Order Item Quantity") AS total_units_sold,
        ROUND(SUM(f.Sales) * 1.0 / COUNT(DISTINCT f."Order Id"), 2) AS avg_order_value
    FROM orders_fact f
""",

"Q2_top_products": """
    SELECT
        p."Category Name", p."Product Name",
        ROUND(SUM(f.Sales), 2) AS product_sales,
        RANK() OVER (ORDER BY SUM(f.Sales) DESC) AS sales_rank
    FROM orders_fact f
    JOIN dim_product p ON f."Product Card Id" = p."Product Card Id"
    GROUP BY p."Category Name", p."Product Name"
    ORDER BY product_sales DESC
    LIMIT 20
""",

"Q3_region_revenue": """
    SELECT
        "Order Region",
        ROUND(SUM(Sales), 2) AS region_sales,
        ROUND(100.0 * SUM(Sales) / (SELECT SUM(Sales) FROM orders_fact), 2) AS pct_of_total_sales
    FROM orders_fact
    GROUP BY "Order Region"
    ORDER BY region_sales DESC
""",

"Q4_segment_sales": """
    SELECT
        c."Customer Segment",
        ROUND(SUM(f.Sales), 2) AS segment_sales,
        COUNT(DISTINCT f."Order Id") AS segment_orders,
        DENSE_RANK() OVER (ORDER BY SUM(f.Sales) DESC) AS segment_rank
    FROM orders_fact f
    JOIN dim_customer c ON f."Order Customer Id" = c."Customer Id"
    GROUP BY c."Customer Segment"
    ORDER BY segment_sales DESC
""",

"Q5_shipping_mode_mix": """
    SELECT
        "Shipping Mode",
        COUNT(*) AS n_line_items,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders_fact), 2) AS pct_of_line_items
    FROM orders_fact
    GROUP BY "Shipping Mode"
    ORDER BY n_line_items DESC
""",

"Q6_avg_shipping_duration": """
    SELECT
        "Shipping Mode",
        ROUND(AVG("Days for shipping (real)"), 2) AS avg_actual_days,
        ROUND(AVG("Days for shipment (scheduled)"), 2) AS avg_scheduled_days,
        ROUND(AVG("Days for shipping (real)" - "Days for shipment (scheduled)"), 2) AS avg_delay_days
    FROM orders_fact
    GROUP BY "Shipping Mode"
    ORDER BY avg_actual_days DESC
""",

"Q7_pct_late": """
    SELECT
        COUNT(*) AS total_line_items,
        SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) AS late_line_items,
        ROUND(100.0 * SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_late
    FROM orders_fact
""",

"Q8_region_late_rate": """
    SELECT
        "Order Region",
        COUNT(*) AS n_line_items,
        ROUND(100.0 * SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_delivery_rate_pct
    FROM orders_fact
    GROUP BY "Order Region"
    HAVING COUNT(*) > 500
    ORDER BY late_delivery_rate_pct DESC
    LIMIT 10
""",

"Q9_high_sales_late_products": """
    SELECT
        p."Product Name",
        ROUND(SUM(f.Sales), 2) AS total_sales,
        ROUND(100.0 * SUM(CASE WHEN f.Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_rate_pct,
        COUNT(*) AS n_line_items
    FROM orders_fact f
    JOIN dim_product p ON f."Product Card Id" = p."Product Card Id"
    GROUP BY p."Product Name"
    HAVING total_sales > (
        SELECT AVG(prod_sales) FROM (
            SELECT SUM(Sales) AS prod_sales FROM orders_fact GROUP BY "Product Card Id"
        )
    )
    AND late_rate_pct > 50
    ORDER BY total_sales DESC
    LIMIT 15
""",

"Q10_mom_growth": """
    WITH monthly_sales AS (
        SELECT strftime('%Y-%m', "order date (DateOrders)") AS ym,
               SUM(Sales) AS total_sales
        FROM orders_fact
        GROUP BY ym
    )
    SELECT
        ym AS order_month,
        ROUND(total_sales, 2) AS total_sales,
        ROUND(LAG(total_sales) OVER (ORDER BY ym), 2) AS prev_month_sales,
        ROUND(100.0 * (total_sales - LAG(total_sales) OVER (ORDER BY ym))
              / LAG(total_sales) OVER (ORDER BY ym), 2) AS mom_growth_pct
    FROM monthly_sales
    ORDER BY ym
""",

"Q11_top3_products_per_category": """
    WITH product_sales AS (
        SELECT p."Category Name", p."Product Name", SUM(f.Sales) AS total_sales
        FROM orders_fact f
        JOIN dim_product p ON f."Product Card Id" = p."Product Card Id"
        GROUP BY p."Category Name", p."Product Name"
    ),
    ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY "Category Name" ORDER BY total_sales DESC) AS rn
        FROM product_sales
    )
    SELECT "Category Name", "Product Name", ROUND(total_sales, 2) AS total_sales, rn
    FROM ranked
    WHERE rn <= 3
    ORDER BY "Category Name", rn
""",

"Q12_cumulative_revenue": """
    WITH monthly_sales AS (
        SELECT strftime('%Y-%m', "order date (DateOrders)") AS ym,
               SUM(Sales) AS total_sales
        FROM orders_fact
        GROUP BY ym
    )
    SELECT
        ym AS order_month,
        ROUND(total_sales, 2) AS total_sales,
        ROUND(SUM(total_sales) OVER (ORDER BY ym
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_sales,
        ROUND(LEAD(total_sales) OVER (ORDER BY ym), 2) AS next_month_sales
    FROM monthly_sales
    ORDER BY ym
""",
}

summary = {}
for name, q in queries.items():
    result = pd.read_sql_query(q, conn)
    result.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    summary[name] = len(result)
    print(f"{name}: {len(result)} rows -> {OUT_DIR}/{name}.csv")

conn.close()
print("\nAll 12 SQL queries executed successfully.")
