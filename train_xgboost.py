# ==========================================================
# train_xgboost.py
#
# Phase-6
# XGBoost Training
#
# Input:
# processed/
#   X_train.csv
#   X_test.csv
#   y_train.csv
#   y_test.csv
#
# Output:
# models/
#   xgboost_model.pkl
#
# results/
#   classification_report.txt
#   evaluation_metrics.txt
#   confusion_matrix.png
#   feature_importance.png
#
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------

INPUT_DIR = "processed"

MODEL_DIR = "models"

RESULT_DIR = "results"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("Loading Processed Dataset")
print("=" * 60)

X_train = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "X_train.csv"
    )
)

X_test = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "X_test.csv"
    )
)

y_train = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "y_train.csv"
    )
)

y_test = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "y_test.csv"
    )
)

# Convert to 1-D array

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()

print()

print("Training Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)

print()

print("Training Labels")

print(pd.Series(y_train).value_counts().sort_index())

print()

print("Testing Labels")

print(pd.Series(y_test).value_counts().sort_index())

# ----------------------------------------------------------
# Create XGBoost Model
# ----------------------------------------------------------

print()
print("=" * 60)
print("Creating XGBoost Model")
print("=" * 60)

model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    n_estimators=50,

    learning_rate=0.0001,

    max_depth=8,

    min_child_weight=2,

    colsample_bytree=0.8,

    gamma=0.001,

    reg_alpha=0.1,

    reg_lambda=1.0,

    tree_method="hist",

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1

)

print()

print(model)

# ----------------------------------------------------------
# Train XGBoost Model
# ----------------------------------------------------------

print()
print("=" * 60)
print("Training XGBoost Model")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print()
print("Model Training Completed Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

print()
print("=" * 60)
print("Making Predictions")
print("=" * 60)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

print()
print("Prediction Completed")

# ----------------------------------------------------------
# Model Evaluation
# ----------------------------------------------------------

print()
print("=" * 60)
print("Evaluating Model")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ----------------------------------------------------------
# Save Evaluation Metrics
# ----------------------------------------------------------

metrics_file = os.path.join(
    RESULT_DIR,
    "evaluation_metrics.txt"
)

with open(metrics_file, "w") as f:

    f.write("=" * 50 + "\n")
    f.write("XGBOOST MODEL EVALUATION\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Accuracy  : {accuracy:.6f}\n")
    f.write(f"Precision : {precision:.6f}\n")
    f.write(f"Recall    : {recall:.6f}\n")
    f.write(f"F1 Score  : {f1:.6f}\n")

print()
print("Evaluation Metrics Saved")

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Healthy",
        "At_Risk",
        "Diseased"
    ]
)

report_file = os.path.join(
    RESULT_DIR,
    "classification_report.txt"
)

with open(report_file, "w") as f:
    f.write(report)

print()
print("Classification Report Saved")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

print()
print("=" * 60)
print("Generating Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "At_Risk",
        "Diseased"
    ]
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion Matrix Saved")

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

print()
print("=" * 60)
print("Generating Feature Importance")
print("=" * 60)

importance = model.feature_importances_

feature_names = X_train.columns

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "feature_importance.csv"
    ),

    index=False

)

plt.figure(figsize=(10,7))

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULT_DIR,
        "feature_importance.png"
    ),

    dpi=300

)

plt.close()

print("Feature Importance Saved")

# ----------------------------------------------------------
# Save Trained Model
# ----------------------------------------------------------

joblib.dump(

    model,

    os.path.join(
        MODEL_DIR,
        "xgboost_model.pkl"
    )

)

print()

print("Model Saved Successfully")

print()

print(os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
))

# ----------------------------------------------------------
# Training Summary
# ----------------------------------------------------------

print()
print("=" * 60)
print("Training Summary")
print("=" * 60)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print()

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ----------------------------------------------------------
# Top 10 Important Features
# ----------------------------------------------------------

print()
print("=" * 60)
print("Top 10 Important Features")
print("=" * 60)

top_features = importance_df.head(10)

for i, row in enumerate(top_features.itertuples(index=False), start=1):

    print(
        f"{i:02d}. "
        f"{row.Feature:<25} "
        f"{row.Importance:.6f}"
    )

# ----------------------------------------------------------
# Prediction Distribution
# ----------------------------------------------------------

print()
print("=" * 60)
print("Prediction Distribution")
print("=" * 60)

prediction_distribution = pd.Series(
    y_pred,
    name="Prediction"
).value_counts().sort_index()

prediction_distribution.index = [
    "Healthy",
    "At_Risk",
    "Diseased"
]

print(prediction_distribution)

# ----------------------------------------------------------
# Actual Distribution
# ----------------------------------------------------------

print()
print("=" * 60)
print("Actual Distribution")
print("=" * 60)

actual_distribution = pd.Series(
    y_test,
    name="Actual"
).value_counts().sort_index()

actual_distribution.index = [
    "Healthy",
    "At_Risk",
    "Diseased"
]

print(actual_distribution)

# ----------------------------------------------------------
# Save Feature Importance Table
# ----------------------------------------------------------

importance_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "feature_importance_rank.csv"
    ),

    index=False

)

# ----------------------------------------------------------
# Final Message
# ----------------------------------------------------------

print()
print("=" * 60)
print("XGBoost Training Completed Successfully")
print("=" * 60)

print()

print("Generated Files")

print("------------------------------------------")

print("Model")

print("  models/xgboost_model.pkl")

print()

print("Evaluation")

print("  results/evaluation_metrics.txt")
print("  results/classification_report.txt")

print()

print("Visualization")

print("  results/confusion_matrix.png")
print("  results/feature_importance.png")
print("  results/feature_importance.csv")
print("  results/feature_importance_rank.csv")

print()

print("Model is Ready for Prediction and SHAP XAI.")
