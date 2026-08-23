"""
Optional Phase - FastAPI service exposing ML predictions and analytical lookups.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Endpoints (per Section 20 of the project guidelines):
    POST /predict-delay        - late-delivery risk for a hypothetical order
    GET  /forecast/{category}  - demand forecast summary (from saved results)
    GET  /product/{product_name} - product performance lookup
    GET  /region/{region}      - regional logistics KPIs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Supply Chain & Logistics Intelligence API",
    description="Serves the late-delivery ML model and validated analytical KPIs.",
    version="1.0.0",
)

_model = None
_df = None
_forecast = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(os.path.join(BASE, "models/late_delivery_best_model.joblib"))
    return _model


def get_data():
    global _df
    if _df is None:
        _df = pd.read_parquet(os.path.join(BASE, "data/processed/dataco_clean.parquet"))
    return _df


def get_forecast():
    global _forecast
    if _forecast is None:
        with open(os.path.join(BASE, "reports/forecast_results.json")) as f:
            _forecast = json.load(f)
    return _forecast


class OrderInput(BaseModel):
    days_for_shipment_scheduled: int
    sales: float
    order_item_quantity: int
    order_item_discount_rate: float
    product_price: float
    shipping_mode: str
    order_region: str
    market: str
    category_name: str
    department_name: str
    customer_segment: str
    type: str
    order_dow: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Supply Chain & Logistics Intelligence API"}


@app.post("/predict-delay")
def predict_delay(order: OrderInput):
    model = get_model()
    row = pd.DataFrame([{
        "Days for shipment (scheduled)": order.days_for_shipment_scheduled,
        "Sales": order.sales,
        "Order Item Quantity": order.order_item_quantity,
        "Order Item Discount Rate": order.order_item_discount_rate,
        "Product Price": order.product_price,
        "Shipping Mode": order.shipping_mode,
        "Order Region": order.order_region,
        "Market": order.market,
        "Category Name": order.category_name,
        "Department Name": order.department_name,
        "Customer Segment": order.customer_segment,
        "Type": order.type,
        "order_dow": order.order_dow,
    }])
    proba = float(model.predict_proba(row)[0, 1])
    return {
        "late_delivery_probability": round(proba, 4),
        "predicted_late": bool(proba >= 0.5),
    }


@app.get("/forecast/{category}")
def forecast(category: str):
    # The saved forecast is at the overall-demand level (Phase H). Category-
    # level forecasts would require re-fitting per category; this endpoint
    # returns the overall validated forecast and is documented as such.
    fc = get_forecast()
    return {
        "requested_category": category,
        "note": "Forecast is at overall monthly-demand level (see reports/forecast_results.json).",
        "selected_model": fc["selected_model"],
        "next_6_month_forecast": fc["next_6_month_forecast"],
    }


@app.get("/product/{product_name}")
def product_performance(product_name: str):
    df = get_data()
    subset = df[df["Product Name"].str.lower() == product_name.lower()]
    if subset.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    return {
        "product_name": product_name,
        "total_sales": round(float(subset["Sales"].sum()), 2),
        "total_units_sold": int(subset["Order Item Quantity"].sum()),
        "avg_delivery_days": round(float(subset["Days for shipping (real)"].mean()), 2),
        "late_delivery_rate_pct": round(100 * float(subset["Late_delivery_risk"].mean()), 2),
    }


@app.get("/region/{region}")
def region_kpis(region: str):
    df = get_data()
    subset = df[df["Order Region"].str.lower() == region.lower()]
    if subset.empty:
        raise HTTPException(status_code=404, detail="Region not found")
    return {
        "region": region,
        "total_sales": round(float(subset["Sales"].sum()), 2),
        "total_orders": int(subset["Order Id"].nunique()),
        "late_delivery_rate_pct": round(100 * float(subset["Late_delivery_risk"].mean()), 2),
        "avg_delivery_days": round(float(subset["Days for shipping (real)"].mean()), 2),
    }
