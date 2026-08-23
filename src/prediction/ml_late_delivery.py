"""
Phase I - Machine Learning: Late Delivery Prediction.

LEAKAGE PREVENTION (hard requirement):
Only features known at ORDER TIME are used. Excluded: Days for shipping (real),
Delivery Status, shipping_delay_days, shipping date, Order Status (post-hoc),
Order Profit Per Order / Item Profit Ratio (realized after fulfillment) - all
of these are only known after the order is placed/fulfilled and would leak
the outcome.

Included (available at order time): Shipping Mode chosen, Days for shipment
(scheduled), Order Region/Market, Category/Department, order value, quantity,
discount, Customer Segment, order timing (day of week, month).
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/dataco_clean.parquet")

TARGET = "Late_delivery_risk"

PRE_SHIPMENT_NUMERIC = [
    "Days for shipment (scheduled)", "Sales", "Order Item Quantity",
    "Order Item Discount Rate", "Product Price",
]
PRE_SHIPMENT_CATEGORICAL = [
    "Shipping Mode", "Order Region", "Market", "Category Name",
    "Department Name", "Customer Segment", "Type", "order_dow",
]
FEATURES = PRE_SHIPMENT_NUMERIC + PRE_SHIPMENT_CATEGORICAL

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), PRE_SHIPMENT_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), PRE_SHIPMENT_CATEGORICAL),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
roc_data = {}
best_model_name, best_auc, best_pipeline = None, -1, None

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    results[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(name, "-> ROC-AUC:", round(auc, 4), "F1:", round(results[name]["f1_score"], 4))

    if auc > best_auc:
        best_auc, best_model_name, best_pipeline = auc, name, pipe

results["best_model"] = best_model_name
results["leakage_prevention_note"] = (
    "Only pre-shipment (order-time) features were used. Fields only known "
    "after fulfillment (actual shipping days, delivery status, shipping "
    "date, realized profit) were excluded to prevent target leakage."
)
results["baseline_majority_class_accuracy"] = round(max(y.mean(), 1 - y.mean()), 4)

with open("reports/ml_model_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save best model
joblib.dump(best_pipeline, "models/late_delivery_best_model.joblib")

# ROC curve comparison plot
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Late Delivery Prediction Models")
ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/16_roc_curves.png", bbox_inches="tight")
plt.close(fig)

# Feature importance for best tree-based model (if applicable)
if best_model_name in ("Random Forest", "Gradient Boosting"):
    prep = best_pipeline.named_steps["prep"]
    feat_names = (PRE_SHIPMENT_NUMERIC +
                  list(prep.named_transformers_["cat"].get_feature_names_out(PRE_SHIPMENT_CATEGORICAL)))
    importances = best_pipeline.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
        .sort_values("importance", ascending=False).head(15)
    imp_df.to_csv("reports/ml_feature_importance_top15.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#2c6fbb")
    ax.set_title(f"Top 15 Feature Importances - {best_model_name}")
    fig.tight_layout()
    fig.savefig("reports/figures/17_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

print("\nBest model:", best_model_name, "ROC-AUC:", round(best_auc, 4))
