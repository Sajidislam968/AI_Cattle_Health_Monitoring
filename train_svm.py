# ==========================================================
# train_svm.py
#
# Phase-6
# SVM Training
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
#   svm_model.pkl
#
# results/
#   svm_evaluation_metrics.txt
#   svm_classification_report.txt
#   svm_confusion_matrix.png
#   svm_feature_importance.csv
#   svm_feature_importance.png
#   svm_feature_importance_rank.csv
#
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance


# ==========================================================
# Directories
# ==========================================================

INPUT_DIR = "processed"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Processed Dataset")
print("=" * 60)

X_train = pd.read_csv(
    os.path.join(INPUT_DIR, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(INPUT_DIR, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(INPUT_DIR, "y_train.csv")
)

y_test = pd.read_csv(
    os.path.join(INPUT_DIR, "y_test.csv")
)


# ==========================================================
# Convert Labels to 1-D
# ==========================================================

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


print()

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print()

print("Training Labels")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print()

print("Testing Labels")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)


# ==========================================================
# Create SVM Model
# ==========================================================

print()
print("=" * 60)
print("Creating SVM Model")
print("=" * 60)


model = SVC(

    C=1.0,

    kernel="rbf",

    gamma="scale",

    probability=True,

    class_weight="balanced",

    random_state=42
)


print()
print(model)


# ==========================================================
# Train SVM Model
# ==========================================================

print()
print("=" * 60)
print("Training SVM Model")
print("=" * 60)

print()
print("SVM training may take some time...")
print()

model.fit(
    X_train,
    y_train
)

print()
print("Model Training Completed Successfully")


# ==========================================================
# Prediction
# ==========================================================

print()
print("=" * 60)
print("Making Predictions")
print("=" * 60)

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)

print()
print("Prediction Completed")


# ==========================================================
# Model Evaluation
# ==========================================================

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
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ==========================================================
# Save Evaluation Metrics
# ==========================================================

metrics_file = os.path.join(
    RESULT_DIR,
    "svm_evaluation_metrics.txt"
)


with open(metrics_file, "w") as f:

    f.write("=" * 50 + "\n")
    f.write("SVM MODEL EVALUATION\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Accuracy  : {accuracy:.6f}\n"
    )

    f.write(
        f"Precision : {precision:.6f}\n"
    )

    f.write(
        f"Recall    : {recall:.6f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.6f}\n"
    )


print()
print("Evaluation Metrics Saved")


# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(

    y_test,

    y_pred,

    target_names=[
        "Healthy",
        "At_Risk",
        "Diseased"
    ],

    zero_division=0
)


report_file = os.path.join(
    RESULT_DIR,
    "svm_classification_report.txt"
)


with open(report_file, "w") as f:

    f.write(report)


print()
print("Classification Report Saved")


# ==========================================================
# Confusion Matrix
# ==========================================================

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


fig, ax = plt.subplots(
    figsize=(8, 6)
)


disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)


plt.title(
    "SVM Confusion Matrix"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "svm_confusion_matrix.png"
    ),

    dpi=300
)


plt.close()


print(
    "Confusion Matrix Saved"
)


# ==========================================================
# Permutation Feature Importance
# ==========================================================

print()
print("=" * 60)
print("Generating SVM Feature Importance")
print("=" * 60)

print()
print(
    "Calculating permutation importance..."
)


# ----------------------------------------------------------
# Use a subset for faster calculation
# ----------------------------------------------------------

importance_sample_size = min(
    20000,
    len(X_test)
)


X_importance = X_test.iloc[
    :importance_sample_size
]

y_importance = y_test[
    :importance_sample_size
]


perm_result = permutation_importance(

    model,

    X_importance,

    y_importance,

    n_repeats=5,

    random_state=42,

    scoring="accuracy",

    n_jobs=-1
)


# ==========================================================
# Create Importance DataFrame
# ==========================================================

importance_df = pd.DataFrame({

    "Feature":
        X_train.columns,

    "Importance":
        perm_result.importances_mean,

    "Importance_STD":
        perm_result.importances_std

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False
)


# ==========================================================
# Save Feature Importance
# ==========================================================

importance_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "svm_feature_importance.csv"
    ),

    index=False
)


print()
print(
    "Feature Importance CSV Saved"
)


# ==========================================================
# Feature Importance Plot
# ==========================================================

plt.figure(
    figsize=(10, 7)
)


plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]
)


plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "SVM Feature Importance"
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_DIR,
        "svm_feature_importance.png"
    ),

    dpi=300
)


plt.close()


print(
    "Feature Importance Plot Saved"
)


# ==========================================================
# Save Ranked Feature Importance
# ==========================================================

importance_df.to_csv(

    os.path.join(
        RESULT_DIR,
        "svm_feature_importance_rank.csv"
    ),

    index=False
)


# ==========================================================
# Save Trained Model
# ==========================================================

model_path = os.path.join(

    MODEL_DIR,

    "svm_model.pkl"
)


joblib.dump(

    model,

    model_path
)


print()
print(
    "Model Saved Successfully"
)

print()
print(model_path)


# ==========================================================
# Training Summary
# ==========================================================

print()
print("=" * 60)
print("Training Summary")
print("=" * 60)


print(
    f"Training Samples : {len(X_train)}"
)

print(
    f"Testing Samples  : {len(X_test)}"
)

print()

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ==========================================================
# Top 10 Important Features
# ==========================================================

print()
print("=" * 60)
print("Top 10 Important Features")
print("=" * 60)


top_features = importance_df.head(10)


for i, row in enumerate(

    top_features.itertuples(
        index=False
    ),

    start=1

):

    print(

        f"{i:02d}. "
        f"{row.Feature:<25} "
        f"{row.Importance:.6f}"

    )


# ==========================================================
# Prediction Distribution
# ==========================================================

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


print(
    prediction_distribution
)


# ==========================================================
# Actual Distribution
# ==========================================================

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


print(
    actual_distribution
)


# ==========================================================
# Final Message
# ==========================================================

print()
print("=" * 60)
print("SVM Training Completed Successfully")
print("=" * 60)

print()

print("Generated Files")

print("------------------------------------------")

print("Model")

print(
    "  models/svm_model.pkl"
)

print()

print("Evaluation")

print("  results/svm_evaluation_metrics.txt")
print("  results/svm_classification_report.txt")

print()

print("Visualization")

print("  results/svm_confusion_matrix.png")

print("  results/svm_feature_importance.png")

print("  results/svm_feature_importance.csv")

print("  results/svm_feature_importance_rank.csv")

print()

print("SVM Model is Ready for Comparison and XAI.")