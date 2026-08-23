# Power BI Dashboard Build Guide

Source file: `dashboard/PowerBI_DataCo_Extract.xlsx`
Sheets: `Fact_Orders` (180,519 rows, one per order line item), `Dim_Product`,
`Dim_Customer`, `Monthly_KPI` (pre-aggregated).

## 1. Import & Model
1. Power BI Desktop -> Get Data -> Excel -> select `PowerBI_DataCo_Extract.xlsx` -> load all 4 sheets.
2. Model view: create relationships
   - `Fact_Orders[Product Card Id]` -> `Dim_Product[Product Card Id]` (many-to-one)
   - `Fact_Orders[Order Customer Id]` -> `Dim_Customer[Order Customer Id]` (many-to-one)
3. Create a Date table (Modeling -> New Table):
   `DateTable = CALENDAR(MIN(Fact_Orders[order date (DateOrders)]), MAX(Fact_Orders[order date (DateOrders)]))`
   Mark it as a Date Table, relate to `Fact_Orders[order date (DateOrders)]`.

## 2. Core DAX Measures
```
Total Revenue = SUM(Fact_Orders[Sales])

Total Orders = DISTINCTCOUNT(Fact_Orders[Order Id])

Total Units Sold = SUM(Fact_Orders[Order Item Quantity])

Avg Order Value = DIVIDE([Total Revenue], [Total Orders])

Late Delivery Rate % =
DIVIDE(
    CALCULATE(COUNTROWS(Fact_Orders), Fact_Orders[Late_delivery_risk] = 1),
    COUNTROWS(Fact_Orders)
)

On-Time Delivery Rate % = 1 - [Late Delivery Rate %]

Avg Delivery Days = AVERAGE(Fact_Orders[Days for shipping (real)])

Avg Scheduled Days = AVERAGE(Fact_Orders[Days for shipment (scheduled)])

Avg Delay Days = AVERAGE(Fact_Orders[shipping_delay_days])

Cancellation Rate % =
DIVIDE(
    CALCULATE(DISTINCTCOUNT(Fact_Orders[Order Id]), Fact_Orders[Order Status] = "CANCELED"),
    [Total Orders]
)

MoM Revenue Growth % =
VAR CurrMonth = [Total Revenue]
VAR PrevMonth = CALCULATE([Total Revenue], DATEADD(DateTable[Date], -1, MONTH))
RETURN DIVIDE(CurrMonth - PrevMonth, PrevMonth)
```

## 3. Pages to Build (per Section 19 of the guidelines)
1. **Executive Overview** - Total Revenue, Total Orders, Total Units, On-Time
   Delivery Rate cards; monthly revenue trend line; region map/bar.
2. **Sales & Product Analytics** - Top categories bar, top products table,
   customer segment pie, MoM growth line.
3. **Supplier Performance** - table from `reports/department_supplier_scorecard.csv`
   (Department Name used as documented supplier proxy) with revenue, late
   rate, volume, and a slicer.
4. **Logistics Dashboard** - shipping mode bar, delay-days box/violin (or
   average bar as PBI has no native boxplot without a custom visual), late
   rate by region map.
5. **Inventory/Demand Dashboard** - import `reports/inventory_demand_profile.csv`
   as a 5th table; fast/slow mover bar, demand variability scatter, forecast
   line from `reports/forecast_results.json` (paste the 6 forecast points as
   a small manual table since Power BI cannot read JSON without Power Query
   M transformation - a `Forecast` sheet added to the Excel extract works too).
6. **ML Insights** - import `reports/ml_model_comparison.json` metrics as a
   small table (model name, accuracy, precision, recall, F1, ROC-AUC) plus
   the feature-importance chart image (`reports/figures/17_feature_importance.png`)
   inserted as a static image, or rebuilt as a PBI bar from
   `reports/ml_feature_importance_top15.csv`.

## 4. Formatting
- Use a consistent 4-color palette (blue #2c6fbb, green #3f8f5f, orange
  #b3763b, red #c0392b - matches the Python chart palette used throughout
  this project for visual consistency between the PDF report and dashboard).
- Number format: currency `$#,##0`, percentages `0.0%`.
- Add a text box on the Supplier and Inventory pages stating the documented
  proxy-data limitations (Department Name as supplier proxy; demand-based
  indicators, not true stock levels) so dashboard users see the same
  disclosure as the written report.

## 5. Publish
File -> Publish -> select workspace, or File -> Export -> Power BI Template
(.pbit) to share the model/measures without the data if data privacy is a
concern for submission.
