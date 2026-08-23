"""
Generates the final capstone PDF report from all validated /reports outputs.
"""
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

FIG = "reports/figures"
OUT = "reports/Supply_Chain_Intelligence_Project_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=10, textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="SubTitle", fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="H1", fontSize=17, leading=22, spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2c6fbb")))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#666666"), spaceAfter=14))
styles.add(ParagraphStyle(name="KPI", fontSize=10.5, leading=15))
styles.add(ParagraphStyle(name="Note", fontSize=9, leading=13, textColor=colors.HexColor("#7a4a00"), backColor=colors.HexColor("#fff6e0"), borderPadding=6, spaceAfter=10))

story = []

def h1(text): story.append(Paragraph(text, styles["H1"]))
def h2(text): story.append(Paragraph(text, styles["H2"]))
def body(text): story.append(Paragraph(text, styles["Body"]))
def note(text): story.append(Paragraph("<b>Limitation / Note:</b> " + text, styles["Note"]))
def img(path, width=6.4*inch, caption=None):
    story.append(Image(path, width=width, height=width*0.56))
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Spacer(1, 10))

def kpi_table(rows, col_widths=(3.2*inch, 2.8*inch)):
    data = [["Metric", "Value"]] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

def df_table(df, col_widths=None, max_rows=15):
    df = df.head(max_rows)
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c6fbb")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

# ============================================================================
# COVER PAGE
# ============================================================================
story.append(Spacer(1, 1.6*inch))
story.append(Paragraph("Supply Chain & Logistics Intelligence System", styles["TitleBig"]))
story.append(Spacer(1, 8))
story.append(Paragraph("A Data Analytics, Forecasting & Machine Learning Capstone Project", styles["SubTitle"]))
story.append(Spacer(1, 6))
story.append(Paragraph("Dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)", styles["SubTitle"]))
story.append(Paragraph("180,519 order-line-item records | January 2015 - January 2018", styles["SubTitle"]))
story.append(Spacer(1, 1.5*inch))

kpis = json.load(open("reports/kpi_summary.json"))
cover_rows = [
    ["Total Revenue", f"${kpis['total_revenue']:,.2f}"],
    ["Total Orders", f"{kpis['total_orders']:,}"],
    ["Average Order Value", f"${kpis['average_order_value']:,.2f}"],
    ["On-Time Delivery Rate", f"{kpis['on_time_delivery_rate_pct']}%"],
    ["Late-Delivery Rate", f"{kpis['late_delivery_rate_pct']}%"],
]
kpi_table(cover_rows)
story.append(PageBreak())

# ============================================================================
# TABLE OF CONTENTS (simple)
# ============================================================================
h1("Table of Contents")
toc_items = [
    "1. Executive Summary", "2. Business Problem & Objectives",
    "3. Data Understanding & Cleaning (Phase A)", "4. SQL Analysis (Phase B)",
    "5. Supply Chain KPIs (Phase C)", "6. Exploratory Data Analysis (Phase D)",
    "7. Supplier Performance Analysis (Phase E)", "8. Inventory / Demand Analytics (Phase F)",
    "9. Statistical Testing (Phase G)", "10. Demand Forecasting (Phase H)",
    "11. Machine Learning: Late-Delivery Prediction (Phase I)",
    "12. Power BI Dashboard & Optional Components (FastAPI, GenAI)",
    "13. Limitations, Assumptions & Recommendations", "14. Appendix",
]
story.append(ListFlowable([ListItem(Paragraph(t, styles["Body"])) for t in toc_items], bulletType="bullet"))
story.append(PageBreak())

# ============================================================================
# 1. EXECUTIVE SUMMARY
# ============================================================================
h1("1. Executive Summary")
exec_summary = open("reports/genai_executive_summary.txt").read()
for para in exec_summary.split("\n\n"):
    para = para.replace("\n", " ").strip()
    if para:
        body(para)
story.append(PageBreak())

# ============================================================================
# 2. BUSINESS PROBLEM
# ============================================================================
h1("2. Business Problem & Objectives")
body("A supply-chain organization needs visibility across orders, shipping, and "
     "delivery performance to identify inefficiencies, reduce delays, and improve "
     "logistics performance. This project builds an end-to-end analytics system "
     "spanning data cleaning, SQL analysis, exploratory data analysis, KPI "
     "engineering, statistical testing, demand forecasting, and machine-learning "
     "prediction of late deliveries, culminating in a Power BI-ready dataset and "
     "this report.")
h2("Main Objectives")
objectives = [
    "Understand and clean supply-chain data at the correct grain.",
    "Use SQL (window functions, CTEs, conditional aggregation) for operational analysis.",
    "Perform EDA and identify operational bottlenecks.",
    "Create business-defined logistics and inventory KPIs.",
    "Analyze supplier (proxy) and delivery performance.",
    "Build and validate a demand-forecasting model.",
    "Build and compare machine-learning models for late-delivery prediction.",
    "Prepare a Power BI-ready data extract, and optionally expose predictions via FastAPI and a GenAI executive summary.",
]
story.append(ListFlowable([ListItem(Paragraph(o, styles["Body"])) for o in objectives], bulletType="bullet"))
story.append(PageBreak())

# ============================================================================
# 3. DATA UNDERSTANDING & CLEANING
# ============================================================================
h1("3. Data Understanding & Cleaning (Phase A)")
clean_log = json.load(open("logs/phase_a_cleaning_log.json"))
body(f"Raw dataset: <b>{clean_log['raw_shape'][0]:,} rows x {clean_log['raw_shape'][1]} columns</b>, "
     f"latin1-encoded. Processed dataset: <b>{clean_log['processed_shape'][0]:,} rows x "
     f"{clean_log['processed_shape'][1]} columns</b> after cleaning and feature engineering.")

h2("Grain Determination")
body(clean_log["grain_check"]["conclusion"])
kpi_table([
    ["Total rows (line items)", f"{clean_log['grain_check']['n_rows']:,}"],
    ["Unique Order IDs", f"{clean_log['grain_check']['n_unique_order_ids']:,}"],
    ["Unique Order Item IDs", f"{clean_log['grain_check']['n_unique_order_item_ids']:,}"],
])

h2("Cleaning Actions Taken")
clean_actions = [
    f"Dropped {len(clean_log['columns_dropped_pii_or_unusable'])} PII / unusable columns: "
    + ", ".join(clean_log["columns_dropped_pii_or_unusable"]),
    f"Exact duplicate rows found and removed: {clean_log['exact_duplicate_rows']}",
    "Order date and shipping date converted to proper datetime types.",
    f"Order Status distribution reviewed; {clean_log['cancelled_or_fraud_rows']:,} cancelled/suspected-fraud "
    "rows flagged with a boolean column (retained, not deleted, since they are real business events).",
    "Categorical fields (Shipping Mode, Market, Region, Segment, etc.) trimmed/standardized.",
    "Outliers in Sales, Quantity, Discount, and Delivery Days flagged via IQR method (not silently dropped).",
]
story.append(ListFlowable([ListItem(Paragraph(a, styles["Body"])) for a in clean_actions], bulletType="bullet"))

note(clean_log["documented_limitations"]["supplier_entity"])
note(clean_log["documented_limitations"]["inventory_stock_levels"])
story.append(PageBreak())

# ============================================================================
# 4. SQL ANALYSIS
# ============================================================================
h1("4. SQL Analysis (Phase B)")
body("Twelve business and operational queries were authored in MySQL syntax "
     "(<font face='Courier'>sql/business_queries.sql</font>) using JOINs against normalized product/customer "
     "dimensions, CTEs, CASE-based conditional aggregation, and window functions "
     "(RANK, DENSE_RANK, ROW_NUMBER, LAG, LEAD, running SUM). Queries were executed "
     "against a SQLite mirror of the cleaned data for local validation; full results "
     "are saved under <font face='Courier'>reports/sql_results/</font>.")

h2("Q1 - Total Sales, Orders, Units Sold")
q1 = pd.read_csv("reports/sql_results/Q1_totals.csv")
df_table(q1)

h2("Q3 - Revenue by Region (Top 8)")
q3 = pd.read_csv("reports/sql_results/Q3_region_revenue.csv")
df_table(q3, max_rows=8)

h2("Q8 - Regions with Highest Late-Delivery Rate")
q8 = pd.read_csv("reports/sql_results/Q8_region_late_rate.csv")
df_table(q8, max_rows=8)

h2("Q10 - Month-over-Month Sales Growth (sample)")
q10 = pd.read_csv("reports/sql_results/Q10_mom_growth.csv")
df_table(q10, max_rows=8)

h2("Q11 - Top 3 Products per Category (sample, ROW_NUMBER window)")
q11 = pd.read_csv("reports/sql_results/Q11_top3_products_per_category.csv")
df_table(q11, max_rows=9)

body("All 12 queries, including Q2 (top products via RANK), Q4 (segment sales via "
     "DENSE_RANK), Q5-Q7 (shipping mode mix, average duration, % late), Q9 (high-sales "
     "products with delivery issues), and Q12 (cumulative revenue via window SUM), "
     "are included in full in <font face='Courier'>reports/sql_results/</font> and "
     "<font face='Courier'>sql/business_queries.sql</font>.")
story.append(PageBreak())

# ============================================================================
# 5. KPIs
# ============================================================================
h1("5. Supply Chain KPIs (Phase C)")
kpi_rows = [
    ["Total Revenue", f"${kpis['total_revenue']:,.2f}"],
    ["Total Orders (distinct)", f"{kpis['total_orders']:,}"],
    ["Total Units Sold", f"{kpis['total_units_sold']:,}"],
    ["Average Order Value", f"${kpis['average_order_value']:,.2f}"],
    ["Average Delivery Time", f"{kpis['average_delivery_time_days']} days"],
    ["On-Time Delivery Rate", f"{kpis['on_time_delivery_rate_pct']}%"],
    ["Late-Delivery Rate", f"{kpis['late_delivery_rate_pct']}%"],
    ["Cancellation Rate", f"{kpis['cancellation_rate_pct']}%"],
    ["Sales Growth (first vs. last complete month)", f"{kpis['sales_growth_rate_pct_first_vs_last_complete_month']}%"],
    ["Top Category", f"{kpis['top_category']} ({kpis['top_category_revenue_share_pct']}% of revenue)"],
    ["Top Region", f"{kpis['top_region']} ({kpis['top_region_revenue_share_pct']}% of revenue)"],
]
kpi_table(kpi_rows)
note(kpis["note_on_avg_shipping_cost"])
story.append(PageBreak())

# ============================================================================
# 6. EDA
# ============================================================================
h1("6. Exploratory Data Analysis (Phase D)")
body("14 charts were produced, each answering a specific business question. "
     "Selected highlights below; the full set is in "
     "<font face='Courier'>reports/figures/</font>.")

img(f"{FIG}/01_monthly_sales_trend.png",
    caption="Figure 1. Monthly sales trend. Note the abrupt volume drop starting Oct 2017 "
            "- confirmed as a data-export truncation artifact, not a real demand collapse "
            "(see Phase H).")
story.append(PageBreak())

img(f"{FIG}/02_top_categories_by_revenue.png", caption="Figure 2. Top 12 product categories by revenue.")
img(f"{FIG}/03_revenue_by_region.png", caption="Figure 3. Revenue distribution by order region.")
story.append(PageBreak())

img(f"{FIG}/05_delivery_days_by_shipping_mode.png",
    caption="Figure 4. Actual delivery days by shipping mode - the visual basis for the ANOVA test in Phase G.")
img(f"{FIG}/08_late_delivery_rate_by_region.png", caption="Figure 5. Regions with the highest late-delivery rates.")
story.append(PageBreak())

img(f"{FIG}/10_correlation_heatmap.png", caption="Figure 6. Correlation among key operational variables.")
img(f"{FIG}/13_late_delivery_trend.png", caption="Figure 7. Late-delivery rate trend over time.")
story.append(PageBreak())

# ============================================================================
# 7. SUPPLIER PERFORMANCE (PROXY)
# ============================================================================
h1("7. Supplier Performance Analysis (Phase E)")
note("This dataset has no dedicated supplier entity/ID. 'Department Name' (the "
     "selling department) is used as a labeled proxy for supplier-level performance "
     "throughout this section. No metric here should be read as true supplier data.")
supplier_df = pd.read_csv("reports/department_supplier_scorecard.csv")
supplier_df.columns = ["Department (Supplier Proxy)", "Revenue", "Late Rate %", "Line Items"]
supplier_df["Revenue"] = supplier_df["Revenue"].round(0).astype(int).apply(lambda x: f"${x:,}")
df_table(supplier_df, max_rows=12)
story.append(PageBreak())

# ============================================================================
# 8. INVENTORY / DEMAND ANALYTICS
# ============================================================================
h1("8. Inventory / Demand Analytics (Phase F)")
inv_summary = json.load(open("reports/inventory_analytics_summary.json"))
note(inv_summary["note"])
kpi_table([
    ["Products profiled", f"{inv_summary['n_products_profiled']}"],
    ["Fast movers", f"{inv_summary['fast_movers_count']}"],
    ["Slow movers", f"{inv_summary['slow_movers_count']}"],
])
h2("Top 5 Fast Movers (avg. units/month)")
fast_rows = [[k, f"{v:.1f}"] for k, v in inv_summary["top_5_fast_movers"].items()]
kpi_table(fast_rows, col_widths=(4*inch, 2*inch))
story.append(PageBreak())

# ============================================================================
# 9. STATISTICS
# ============================================================================
h1("9. Statistical Testing (Phase G)")
stats_ = json.load(open("reports/statistical_test_results.json"))

h2("Hypothesis Test 1: Shipping Mode vs. Delivery Time (ANOVA)")
a = stats_["anova_shipping_mode_vs_delivery_time"]
body(f"<b>H0:</b> {a['H0']}. <b>Result:</b> F = {a['f_statistic']}, p-value &lt; 0.001. "
     f"<b>Conclusion:</b> {a['conclusion']}")

h2("Hypothesis Test 2: Customer Segment vs. Late Delivery (Chi-square)")
c = stats_["chisquare_segment_vs_late_delivery"]
body(f"<b>H0:</b> {c['H0']}. <b>Result:</b> chi-sq = {c['chi2_statistic']}, "
     f"dof = {c['dof']}, p-value = {c['p_value']:.3f}. <b>Conclusion:</b> {c['conclusion']}")

h2("95% Confidence Interval: Mean Shipping Delay")
ci = stats_["confidence_interval_mean_shipping_delay_days"]
body(f"Mean actual-vs-scheduled delay = <b>{ci['mean']} days</b> "
     f"(95% CI: {ci['95pct_ci_lower']} - {ci['95pct_ci_upper']} days).")

h2("Outlier Rates (IQR method)")
out_rows = [[k.replace("outlier_", "").replace("_", " "), f"{v}%"] for k, v in stats_["outlier_rate_pct"].items()]
kpi_table(out_rows, col_widths=(4*inch, 2*inch))
story.append(PageBreak())

# ============================================================================
# 10. FORECASTING
# ============================================================================
h1("10. Demand Forecasting (Phase H)")
fc = json.load(open("reports/forecast_results.json"))
body(f"Monthly unit demand was aggregated and modeled with a time-aware train/test "
     f"split ({fc['train_months']} training months, {fc['test_months']} held-out test "
     f"months). Months {', '.join(fc['excluded_months_data_artifact'])} were excluded "
     f"from training as a confirmed data-truncation artifact (see Figure 1).")

kpi_table([
    ["Baseline (naive) MAE", f"{fc['baseline_naive_mae']} units"],
    ["Baseline (naive) RMSE", f"{fc['baseline_naive_rmse']} units"],
    ["Holt-Winters MAE", f"{fc['holt_winters_mae']} units"],
    ["Holt-Winters RMSE", f"{fc['holt_winters_rmse']} units"],
    ["Improvement over baseline", f"{fc['improvement_over_baseline_pct']}%"],
    ["Selected model", fc["selected_model"]],
])
img(f"{FIG}/15_forecast_actual_vs_predicted.png", caption="Figure 8. Actual vs. Holt-Winters forecast, plus 6-month forward projection.")

h2("6-Month Forward Forecast (Units)")
fc_rows = [[k, f"{v:,.0f}"] for k, v in fc["next_6_month_forecast"].items()]
kpi_table(fc_rows, col_widths=(3*inch, 3*inch))
story.append(PageBreak())

# ============================================================================
# 11. ML
# ============================================================================
h1("11. Machine Learning: Late-Delivery Prediction (Phase I)")
ml = json.load(open("reports/ml_model_comparison.json"))
note(ml["leakage_prevention_note"])
body(f"Baseline majority-class accuracy: <b>{ml['baseline_majority_class_accuracy']*100:.1f}%</b> "
     f"(the naive floor any model must beat).")

h2("Model Comparison")
model_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
for name, m in ml.items():
    if isinstance(m, dict) and "roc_auc" in m:
        model_rows.append([name, f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                            f"{m['recall']:.3f}", f"{m['f1_score']:.3f}", f"{m['roc_auc']:.3f}"])
t = Table(model_rows, colWidths=[1.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.9*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fa")]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("BACKGROUND", (0, [r[0] for r in model_rows].index(ml["best_model"])), (-1, [r[0] for r in model_rows].index(ml["best_model"])), colors.HexColor("#d9ead3")),
]))
story.append(t)
story.append(Spacer(1, 10))
body(f"<b>Best model: {ml['best_model']}</b>, selected on ROC-AUC.")

img(f"{FIG}/16_roc_curves.png", caption="Figure 9. ROC curves for all four candidate models.")
img(f"{FIG}/17_feature_importance.png", caption="Figure 10. Top 15 feature importances for the selected Random Forest model.")
story.append(PageBreak())

# ============================================================================
# 12. DASHBOARD & OPTIONAL COMPONENTS
# ============================================================================
h1("12. Power BI Dashboard & Optional Components")
h2("Power BI")
body("A Power BI-ready Excel extract (<font face='Courier'>dashboard/PowerBI_DataCo_Extract.xlsx</font>) "
     "was built with a fact table, product/customer dimensions, and pre-aggregated "
     "monthly KPI, forecast, ML-comparison, inventory, and supplier-scorecard sheets. "
     "A full build guide with DAX measures for every required page (Executive Overview, "
     "Sales & Product Analytics, Supplier Performance, Logistics, Inventory/Demand, "
     "ML Insights) is provided in <font face='Courier'>dashboard/PowerBI_Build_Guide.md</font>.")

h2("FastAPI Service")
body("A FastAPI application (<font face='Courier'>api/main.py</font>) exposes four "
     "endpoints: <font face='Courier'>POST /predict-delay</font>, "
     "<font face='Courier'>GET /forecast/{category}</font>, "
     "<font face='Courier'>GET /product/{product_id}</font>, and "
     "<font face='Courier'>GET /region/{region}</font>. All four were tested end-to-end "
     "against the trained model and cleaned dataset; example: a Standard Class order to "
     "Western Europe returned a 41.1% predicted late-delivery probability.")

h2("GenAI Executive Assistant")
body("A script (<font face='Courier'>src/genai_executive_assistant.py</font>) converts "
     "the validated JSON outputs above into a plain-language executive summary via an "
     "LLM call, with an explicit instruction to use only the supplied figures and never "
     "invent numbers. The Executive Summary in Section 1 of this report was produced by "
     "this component.")
story.append(PageBreak())

# ============================================================================
# 13. LIMITATIONS & RECOMMENDATIONS
# ============================================================================
h1("13. Limitations, Assumptions & Recommendations")
h2("Documented Limitations")
lim = [
    "No dedicated supplier entity: Department Name used as an explicitly labeled proxy.",
    "No on-hand inventory field: demand-based velocity/variability used instead of true stock levels.",
    "No direct shipping-cost field: shipping cost KPIs could not be computed and are not approximated.",
    "Oct 2017-Jan 2018 volume shows a data-export truncation artifact, excluded from forecast training.",
]
story.append(ListFlowable([ListItem(Paragraph(l, styles["Body"])) for l in lim], bulletType="bullet"))

h2("Recommendations")
rec = [
    "Prioritize a shipping-mode-level SLA review; mode is the statistically dominant driver of delay.",
    "Do not target interventions by customer segment for lateness; segment has no significant effect.",
    "Deploy the Random Forest late-delivery score at order entry to proactively flag at-risk orders.",
    "Treat the 6-month demand outlook (~10,000-11,100 units/month) as roughly flat for planning purposes.",
    "If true supplier and inventory data become available, replace the documented proxies with real fields.",
]
story.append(ListFlowable([ListItem(Paragraph(r, styles["Body"])) for r in rec], bulletType="bullet"))
story.append(PageBreak())

# ============================================================================
# 14. APPENDIX
# ============================================================================
h1("14. Appendix")
h2("Technology Stack")
stack = [
    "Python (pandas, numpy) - data processing",
    "SQLite (MySQL-syntax queries) - SQL analysis",
    "Matplotlib / Seaborn - EDA visualization",
    "SciPy / Statsmodels - hypothesis testing & Holt-Winters forecasting",
    "Scikit-learn - late-delivery classification (4-model comparison)",
    "FastAPI - optional model-serving API",
    "openpyxl / xlsxwriter - Power BI-ready Excel extract",
    "reportlab - this PDF report",
]
story.append(ListFlowable([ListItem(Paragraph(s, styles["Body"])) for s in stack], bulletType="bullet"))

h2("Repository Structure")
body("See README.md for the full folder structure and reproduction steps. All code, "
     "SQL, saved models, figures, and intermediate JSON/CSV results referenced in this "
     "report are included in the project deliverable.")

doc = SimpleDocTemplate(OUT, pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)
doc.build(story)
print("PDF report written to", OUT)
