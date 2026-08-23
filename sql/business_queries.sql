-- ============================================================================
-- Supply Chain & Logistics Intelligence System
-- Phase B - SQL Business & Operational Analysis  (MySQL syntax)
-- Schema: orders_fact (one row = one order LINE ITEM)
--         dim_product  (Product Card Id, Product Name, Category Name, Department Name, Product Price)
--         dim_customer (Customer Id, Customer Segment, Customer Country, Customer State, Customer City)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q1. Total sales, total DISTINCT orders, total units sold
--     (uses conditional aggregation grain fix: COUNT(DISTINCT Order Id))
-- ----------------------------------------------------------------------------
SELECT
    ROUND(SUM(f.Sales), 2)                AS total_sales,
    COUNT(DISTINCT f.`Order Id`)          AS total_orders,
    SUM(f.`Order Item Quantity`)          AS total_units_sold,
    ROUND(SUM(f.Sales) / COUNT(DISTINCT f.`Order Id`), 2) AS avg_order_value
FROM orders_fact f;

-- ----------------------------------------------------------------------------
-- Q2. Which products/categories generate the highest sales? (JOIN + RANK)
-- ----------------------------------------------------------------------------
SELECT
    p.`Category Name`,
    p.`Product Name`,
    ROUND(SUM(f.Sales), 2) AS product_sales,
    RANK() OVER (ORDER BY SUM(f.Sales) DESC) AS sales_rank
FROM orders_fact f
JOIN dim_product p ON f.`Product Card Id` = p.`Product Card Id`
GROUP BY p.`Category Name`, p.`Product Name`
ORDER BY product_sales DESC
LIMIT 20;

-- ----------------------------------------------------------------------------
-- Q3. Which regions contribute the most revenue?
-- ----------------------------------------------------------------------------
SELECT
    `Order Region`,
    ROUND(SUM(Sales), 2) AS region_sales,
    ROUND(100.0 * SUM(Sales) / (SELECT SUM(Sales) FROM orders_fact), 2) AS pct_of_total_sales
FROM orders_fact
GROUP BY `Order Region`
ORDER BY region_sales DESC;

-- ----------------------------------------------------------------------------
-- Q4. Which customer segments generate the most sales? (JOIN + DENSE_RANK)
-- ----------------------------------------------------------------------------
SELECT
    c.`Customer Segment`,
    ROUND(SUM(f.Sales), 2) AS segment_sales,
    COUNT(DISTINCT f.`Order Id`) AS segment_orders,
    DENSE_RANK() OVER (ORDER BY SUM(f.Sales) DESC) AS segment_rank
FROM orders_fact f
JOIN dim_customer c ON f.`Order Customer Id` = c.`Customer Id`
GROUP BY c.`Customer Segment`
ORDER BY segment_sales DESC;

-- ----------------------------------------------------------------------------
-- Q5. Which shipping modes are most common?
-- ----------------------------------------------------------------------------
SELECT
    `Shipping Mode`,
    COUNT(*) AS n_line_items,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders_fact), 2) AS pct_of_line_items
FROM orders_fact
GROUP BY `Shipping Mode`
ORDER BY n_line_items DESC;

-- ----------------------------------------------------------------------------
-- Q6. What is the average shipping duration (by shipping mode)?
-- ----------------------------------------------------------------------------
SELECT
    `Shipping Mode`,
    ROUND(AVG(`Days for shipping (real)`), 2)      AS avg_actual_days,
    ROUND(AVG(`Days for shipment (scheduled)`), 2) AS avg_scheduled_days,
    ROUND(AVG(`Days for shipping (real)` - `Days for shipment (scheduled)`), 2) AS avg_delay_days
FROM orders_fact
GROUP BY `Shipping Mode`
ORDER BY avg_actual_days DESC;

-- ----------------------------------------------------------------------------
-- Q7. What percentage of orders are late? (CASE + conditional aggregation)
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_line_items,
    SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) AS late_line_items,
    ROUND(100.0 * SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_late
FROM orders_fact;

-- ----------------------------------------------------------------------------
-- Q8. Which regions have the highest late-delivery rate?
-- ----------------------------------------------------------------------------
SELECT
    `Order Region`,
    COUNT(*) AS n_line_items,
    ROUND(100.0 * SUM(CASE WHEN Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_delivery_rate_pct
FROM orders_fact
GROUP BY `Order Region`
HAVING COUNT(*) > 500
ORDER BY late_delivery_rate_pct DESC
LIMIT 10;

-- ----------------------------------------------------------------------------
-- Q9. Which products have high sales but delivery issues? (JOIN + HAVING)
-- ----------------------------------------------------------------------------
SELECT
    p.`Product Name`,
    ROUND(SUM(f.Sales), 2) AS total_sales,
    ROUND(100.0 * SUM(CASE WHEN f.Late_delivery_risk = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_rate_pct,
    COUNT(*) AS n_line_items
FROM orders_fact f
JOIN dim_product p ON f.`Product Card Id` = p.`Product Card Id`
GROUP BY p.`Product Name`
HAVING total_sales > (SELECT AVG(t.prod_sales) FROM
        (SELECT SUM(Sales) AS prod_sales FROM orders_fact GROUP BY `Product Card Id`) t)
   AND late_rate_pct > 50
ORDER BY total_sales DESC
LIMIT 15;

-- ----------------------------------------------------------------------------
-- Q10. Month-over-month sales growth (CTE + LAG window function)
-- ----------------------------------------------------------------------------
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(`order date (DateOrders)`, '%Y-%m') AS ym,
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
ORDER BY ym;

-- ----------------------------------------------------------------------------
-- Q11. Top 3 products within each category (ROW_NUMBER partitioned window)
-- ----------------------------------------------------------------------------
WITH product_sales AS (
    SELECT
        p.`Category Name`,
        p.`Product Name`,
        SUM(f.Sales) AS total_sales
    FROM orders_fact f
    JOIN dim_product p ON f.`Product Card Id` = p.`Product Card Id`
    GROUP BY p.`Category Name`, p.`Product Name`
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY `Category Name` ORDER BY total_sales DESC) AS rn
    FROM product_sales
)
SELECT `Category Name`, `Product Name`, ROUND(total_sales, 2) AS total_sales, rn
FROM ranked
WHERE rn <= 3
ORDER BY `Category Name`, rn;

-- ----------------------------------------------------------------------------
-- Q12. Cumulative (running total) monthly revenue + next-month lead value
--      (window SUM() OVER for cumulative totals, LEAD for forward look)
-- ----------------------------------------------------------------------------
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(`order date (DateOrders)`, '%Y-%m') AS ym,
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
ORDER BY ym;
