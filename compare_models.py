"""Compare classifiers on the exact split used by train_xgboost.py.

Inputs are the already-preprocessed matrices in processed/. No new split,
resampling, or synthetic data is created here.
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "model_comparison"
MODELS_DIR = PROJECT_ROOT / "models" / "comparison"
CLASS_NAMES = ["Healthy", "At Risk", "Diseased"]
RANDOM_STATE = 42


def load_existing_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the exact persisted train/test split used by XGBoost."""
    x_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0].to_numpy()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0].to_numpy()
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("X_train.csv and X_test.csv have different feature columns.")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    return x_train, x_test, y_train, y_test


def build_models() -> dict[str, Any]:
    """Use the existing XGBoost configuration plus fixed-seed baselines."""
    return {
        "XGBoost": XGBClassifier(
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }


def evaluate_model(model_name: str, model: Any, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    start = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start
    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC_AUC_ovr_weighted": roc_auc_score(y_test, y_probability, multi_class="ovr", average="weighted", labels=[0, 1, 2]),
        "Training_seconds": training_seconds,
    }
    report = str(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))
    (RESULTS_DIR / "classification_reports").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_reports" / f"{model_name.lower().replace(' ', '_')}.txt").write_text(report)
    np.save(RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.npy", y_pred)
    np.save(RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_probabilities.npy", y_probability)
    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    axis.set_title(f"{model_name} confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(3), CLASS_NAMES)
    threshold = confusion.max() / 2
    for row_index in range(3):
        for column_index in range(3):
            axis.text(column_index, row_index, confusion[row_index, column_index], ha="center", va="center", color="white" if confusion[row_index, column_index] > threshold else "black")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close(figure)
    joblib.dump(model, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl")
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_existing_split()
    print(f"Using exact split: X_train={x_train.shape}, X_test={x_test.shape}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    results = []
    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        result = evaluate_model(model_name, model, x_train, x_test, y_train, y_test)
        results.append(result)
        print(f"{model_name}: F1={result['F1_weighted']:.4f}, AUC={result['ROC_AUC_ovr_weighted']:.4f}")
    comparison = pd.DataFrame(results).sort_values("F1_weighted", ascending=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison_metrics.csv", index=False)
    comparison.to_json(RESULTS_DIR / "model_comparison_metrics.json", orient="records", indent=2)
    print("\nModel comparison saved to:")
    print(RESULTS_DIR / "model_comparison_metrics.csv")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
