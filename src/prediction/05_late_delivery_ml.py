"""
Phase I - Machine Learning: Late Delivery Prediction
Predicts whether an order LINE ITEM will be delivered late, using only
features known at ORDER TIME (no post-outcome leakage).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_score, recall_score, f1_score, accuracy_score)
import json

df = pd.read_csv("data/processed/cleaned_supply_chain.csv", parse_dates=["order date (DateOrders)"])

# ---------------------------------------------------------------
# Target: is_late (1 = Late delivery). Exclude cancelled shipments
# (delivery outcome not meaningful for those).
# ---------------------------------------------------------------
model_df = df[df["Delivery Status"] != "Shipping canceled"].copy()

# ---------------------------------------------------------------
# LEAKAGE CHECK: exclude any field only known AFTER the delivery outcome
#   - 'Days for shipping (real)'  -> known only after shipment completes
#   - 'Delivery Status'           -> IS the outcome
#   - 'shipping_delay_days'       -> derived from the real shipping days
#   - 'Late_delivery_risk'        -> pre-computed duplicate of the target
# Only order-time-known fields are kept as features.
# ---------------------------------------------------------------
target = "is_late"
feature_cols_numeric = [
    "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Discount_Rate"
    if "Order Item Discount_Rate" in model_df.columns else "Order Item Discount Rate",
    "Order Item Product Price", "Sales",
]
feature_cols_categorical = [
    "Shipping Mode", "Order Region", "Market", "Category Name",
    "Customer Segment", "Type",
]
model_df["order_dow"] = model_df["order date (DateOrders)"].dt.dayofweek
model_df["order_month_num"] = model_df["order date (DateOrders)"].dt.month
feature_cols_numeric += ["order_dow", "order_month_num"]

X = model_df[feature_cols_numeric + feature_cols_categorical]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
], remainder="passthrough")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
    }
    fitted[name] = pipe

# Baseline: majority-class predictor
baseline_pred = np.full_like(y_test, y_train.mode()[0])
results["Baseline (majority class)"] = {
    "Accuracy": round(accuracy_score(y_test, baseline_pred), 4),
    "Precision": round(precision_score(y_test, baseline_pred, zero_division=0), 4),
    "Recall": round(recall_score(y_test, baseline_pred, zero_division=0), 4),
    "F1": round(f1_score(y_test, baseline_pred, zero_division=0), 4),
    "ROC_AUC": None,
}

best_name = max((k for k in results if k != "Baseline (majority class)"),
                 key=lambda k: results[k]["ROC_AUC"])
best_pipe = fitted[best_name]
best_pred = best_pipe.predict(X_test)
best_proba = best_pipe.predict_proba(X_test)[:, 1]

# Confusion matrix plot
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["On-time", "Late"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["On-time", "Late"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
fig.tight_layout()
fig.savefig("reports/figures/12_confusion_matrix.png", dpi=130)
plt.close(fig)

# ROC curve for all models
fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC_AUC']})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Late Delivery Prediction Models")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/figures/13_roc_curves.png", dpi=130)
plt.close(fig)

# Feature importance (if tree-based best model)
if best_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(feature_cols_categorical))
    all_feature_names = cat_feature_names + feature_cols_numeric
    importances = best_pipe.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.iloc[::-1].plot(kind="barh", ax=ax, color="#2E5EAA")
    ax.set_title(f"Top 15 Feature Importances - {best_name}")
    fig.tight_layout()
    fig.savefig("reports/figures/14_feature_importance.png", dpi=130)
    plt.close(fig)
    top_features = fi.round(4).to_dict()
else:
    top_features = None

output = {
    "target": "is_late (1 = Late delivery, 0 = on-time/advance)",
    "leakage_excluded_fields": [
        "Days for shipping (real)", "Delivery Status", "shipping_delay_days", "Late_delivery_risk"
    ],
    "train_size": len(X_train), "test_size": len(X_test),
    "class_balance_test_set_pct_late": round(100 * y_test.mean(), 2),
    "model_comparison": results,
    "best_model": best_name,
    "classification_report_best_model": classification_report(y_test, best_pred, output_dict=True),
    "top_feature_importances": top_features,
}
with open("reports/phaseI_ml_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

import joblib
joblib.dump(best_pipe, "models/late_delivery_model.pkl")

print("Model comparison:")
for name, m in results.items():
    print(f"  {name}: {m}")
print("\nBest model:", best_name)
print("Saved model to models/late_delivery_model.pkl")
