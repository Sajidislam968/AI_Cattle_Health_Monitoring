# ==========================================================
# shap_xai_engine.py
#
# Phase-7
# SHAP Explainable AI Engine
#
# Input:
# models/xgboost_model.pkl
# processed/X_test.csv
# processed/y_test.csv
#
# Output:
# generated/shap_explanations.csv
# results/shap_summary.png
# results/shap_feature_importance.csv
# results/shap_feature_importance.png
#
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ==========================================================
# Directories
# ==========================================================

MODEL_DIR = "models"
PROCESSED_DIR = "processed"
RESULT_DIR = "results"
GENERATED_DIR = "generated"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    GENERATED_DIR,
    exist_ok=True
)


# ==========================================================
# File Paths
# ==========================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)

X_TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "X_test.csv"
)

Y_TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "y_test.csv"
)

OUTPUT_FILE = os.path.join(
    GENERATED_DIR,
    "shap_explanations.csv"
)


# ==========================================================
# Check Required Files
# ==========================================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"XGBoost model not found: {MODEL_FILE}"
    )


if not os.path.exists(X_TEST_FILE):

    raise FileNotFoundError(
        f"X_test.csv not found: {X_TEST_FILE}"
    )


if not os.path.exists(Y_TEST_FILE):

    raise FileNotFoundError(
        f"y_test.csv not found: {Y_TEST_FILE}"
    )


# ==========================================================
# Load XGBoost Model
# ==========================================================

print("=" * 60)
print("Loading XGBoost Model")
print("=" * 60)

model = joblib.load(
    MODEL_FILE
)

print()

print(
    "Model Loaded Successfully"
)

print(
    "Model Type :",
    type(model).__name__
)


# ==========================================================
# Load Test Dataset
# ==========================================================

print()
print("=" * 60)
print("Loading Test Dataset")
print("=" * 60)


X_test = pd.read_csv(
    X_TEST_FILE
)

y_test = pd.read_csv(
    Y_TEST_FILE
)


# Convert target to 1-D

y_test = y_test.values.ravel()


print()

print(
    "X_test Shape :",
    X_test.shape
)

print(
    "y_test Shape :",
    y_test.shape
)


# ==========================================================
# Verify Feature Compatibility
# ==========================================================

print()
print("=" * 60)
print("Checking Feature Compatibility")
print("=" * 60)


model_features = list(
    X_test.columns
)

print()

print(
    "Number of Features :",
    len(model_features)
)

print()

print("Features:")

for i, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ==========================================================
# Generate Predictions
# ==========================================================

print()
print("=" * 60)
print("Generating Model Predictions")
print("=" * 60)


y_pred = model.predict(
    X_test
)


y_prob = model.predict_proba(
    X_test
)


print()

print(
    "Prediction Completed Successfully"
)


print()

print("Prediction Distribution:")

prediction_distribution = pd.Series(
    y_pred
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
# Create SHAP Explainer
# ==========================================================

print()
print("=" * 60)
print("Creating SHAP Explainer")
print("=" * 60)


explainer = shap.TreeExplainer(
    model
)


print()

print(
    "SHAP TreeExplainer Created Successfully"
)


# ==========================================================
# Initial Message
# ==========================================================

print()
print("=" * 60)
print("SHAP XAI Engine Ready")
print("=" * 60)

print()

print(
    "Loaded Samples :",
    len(X_test)
)

print(
    "Features       :",
    X_test.shape[1]
)

print()

print(
    "Next Step: Calculate SHAP Values"
)

print("=" * 60)

# ==========================================================
# PART 2
# Calculate SHAP Values
# ==========================================================

print()
print("=" * 60)
print("Calculating SHAP Values")
print("=" * 60)


# ----------------------------------------------------------
# SHAP Calculation Sample
# ----------------------------------------------------------
# Full test dataset অনেক বড়।
# তাই প্রথমে controlled sample ব্যবহার করছি।
#
# পরে প্রয়োজন হলে FULL test dataset-এ চালানো যাবে।
# ----------------------------------------------------------

# ----------------------------------------------------------
# Full Test Dataset for SHAP
# ----------------------------------------------------------

print()
print("=" * 60)
print("Preparing Full Test Dataset for SHAP")
print("=" * 60)

# Use the complete test dataset
# This avoids bias from taking only the first 20,000 samples.

X_shap = X_test.copy()

y_shap = y_test.copy()

print()
print("SHAP Samples :", len(X_shap))
print("SHAP Features:", X_shap.shape[1])

print()
print("Full test dataset selected for SHAP.")


print()

print(
    "SHAP Samples :",
    len(X_shap)
)



# ----------------------------------------------------------
# Calculate SHAP Values
# ----------------------------------------------------------

print()
print("Calculating SHAP values for FULL test dataset...")
print("This may take some time.")
print()

shap_values = explainer.shap_values(
    X_shap
)

print()
print("SHAP Value Calculation Completed")

# ==========================================================
# Handle Multiclass SHAP Output
# ==========================================================

print()
print("=" * 60)
print("Processing Multiclass SHAP Values")
print("=" * 60)


print()

print(
    "SHAP Output Type :",
    type(shap_values)
)


# ----------------------------------------------------------
# Different SHAP versions return different formats.
#
# Possible formats:
#
# 1. List:
#    [
#       class_0_values,
#       class_1_values,
#       class_2_values
#    ]
#
# 2. NumPy array:
#    (samples, features, classes)
#
# ----------------------------------------------------------

if isinstance(
    shap_values,
    list
):

    print(
        "Detected SHAP list format"
    )

    print()

    print(
        "Number of Classes :",
        len(shap_values)
    )

    shap_healthy = np.asarray(
        shap_values[0]
    )

    shap_at_risk = np.asarray(
        shap_values[1]
    )

    shap_diseased = np.asarray(
        shap_values[2]
    )


else:

    print(
        "Detected SHAP array format"
    )

    shap_values = np.asarray(
        shap_values
    )

    print()

    print(
        "SHAP Array Shape :",
        shap_values.shape
    )


    # ------------------------------------------------------
    # Expected shape:
    #
    # samples × features × classes
    # ------------------------------------------------------

    if shap_values.ndim == 3:

        shap_healthy = shap_values[
            :, :, 0
        ]

        shap_at_risk = shap_values[
            :, :, 1
        ]

        shap_diseased = shap_values[
            :, :, 2
        ]

    else:

        raise ValueError(
            "Unexpected SHAP value shape: "
            + str(shap_values.shape)
        )


# ==========================================================
# Verify SHAP Shapes
# ==========================================================

print()
print("=" * 60)
print("SHAP Shape Verification")
print("=" * 60)


print()

print(
    "Healthy SHAP Shape :",
    shap_healthy.shape
)

print(
    "At_Risk SHAP Shape :",
    shap_at_risk.shape
)

print(
    "Diseased SHAP Shape :",
    shap_diseased.shape
)


# ==========================================================
# Store SHAP Values
# ==========================================================

print()
print("=" * 60)
print("SHAP Values Ready")
print("=" * 60)


print()

print(
    "Healthy SHAP values ready"
)

print(
    "At_Risk SHAP values ready"
)

print(
    "Diseased SHAP values ready"
)

print()

print(
    "Next Step: Generate Feature Contributions"
)

print("=" * 60)

# ==========================================================
# PART 3
# Generate Feature Contributions
# ==========================================================

print()
print("=" * 60)
print("Generating Feature Contributions")
print("=" * 60)


# ==========================================================
# Feature Names
# ==========================================================

feature_names = list(
    X_shap.columns
)


# ==========================================================
# Human-Readable Feature Names
# ==========================================================

feature_display_names = {

    "Activity_Index":
        "Activity",

    "Lying_Hr":
        "Lying Duration",

    "Rumination_Min":
        "Rumination",

    "Standing_Hr":
        "Standing Duration",

    "Body_Temperature_C":
        "Body Temperature",

    "Walking_km":
        "Walking Activity",

    "Eating_Hr":
        "Feeding",

    "Drinking_Min":
        "Drinking",

    "Heart_Rate_BPM":
        "Heart Rate",

    "THI":
        "Heat Stress",

    "Ambient_Temperature_C":
        "Ambient Temperature",

    "Relative_Humidity":
        "Relative Humidity"
}


# ==========================================================
# Determine Predicted Class
# ==========================================================

predicted_class = model.predict(
    X_shap
)


# ==========================================================
# Prediction Confidence
# ==========================================================

# ==========================================================
# Prediction Probability and Confidence
# ==========================================================

print()
print("=" * 60)
print("Calculating Prediction Probabilities")
print("=" * 60)

prediction_probabilities = model.predict_proba(
    X_shap
)

# Predicted class from highest probability
predicted_class = np.argmax(
    prediction_probabilities,
    axis=1
)

# Confidence = probability of predicted class
confidence_scores = prediction_probabilities[
    np.arange(len(prediction_probabilities)),
    predicted_class
]

print()
print("Prediction Probability Shape :",
      prediction_probabilities.shape)

print("Confidence Calculation Completed")

print()
print("Confidence Statistics:")

print(
    pd.Series(
        confidence_scores,
        name="Confidence_Score"
    ).describe()
)


# ==========================================================
# Class Names
# ==========================================================

class_names = {

    0: "Healthy",

    1: "At_Risk",

    2: "Diseased"
}


predicted_status = [

    class_names[
        int(pred)
    ]

    for pred in predicted_class
]


# ==========================================================
# Initialize Result Lists
# ==========================================================

dominant_factors = []

secondary_factors = []

dominant_contributions = []

secondary_contributions = []


# ==========================================================
# Generate Feature Contributions
# ==========================================================

print()

print(
    "Analyzing feature contributions..."
)


for i in range(
    len(X_shap)
):

    # ------------------------------------------------------
    # Select SHAP values according to predicted class
    # ------------------------------------------------------

    predicted = int(
        predicted_class[i]
    )


    if predicted == 0:

        current_shap = shap_healthy[i]


    elif predicted == 1:

        current_shap = shap_at_risk[i]


    else:

        current_shap = shap_diseased[i]


    # ------------------------------------------------------
    # Absolute contribution
    # ------------------------------------------------------

    absolute_shap = np.abs(
        current_shap
    )


    # ------------------------------------------------------
    # Sort features by contribution
    # ------------------------------------------------------

    sorted_indices = np.argsort(
        absolute_shap
    )[::-1]


    # ------------------------------------------------------
    # Dominant Feature
    # ------------------------------------------------------

    dominant_index = sorted_indices[0]

    dominant_feature = feature_names[
        dominant_index
    ]

    dominant_value = current_shap[
        dominant_index
    ]


    # ------------------------------------------------------
    # Secondary Feature
    # ------------------------------------------------------

    secondary_index = sorted_indices[1]

    secondary_feature = feature_names[
        secondary_index
    ]

    secondary_value = current_shap[
        secondary_index
    ]


    # ------------------------------------------------------
    # Convert to readable names
    # ------------------------------------------------------

    dominant_name = feature_display_names.get(

        dominant_feature,

        dominant_feature
    )


    secondary_name = feature_display_names.get(

        secondary_feature,

        secondary_feature
    )


    # ------------------------------------------------------
    # Store Results
    # ------------------------------------------------------

    dominant_factors.append(
        dominant_name
    )

    secondary_factors.append(
        secondary_name
    )

    dominant_contributions.append(
        round(
            float(dominant_value),
            6
        )
    )

    secondary_contributions.append(
        round(
            float(secondary_value),
            6
        )
    )


print()

print(
    "Feature Contribution Analysis Completed"
)


# ==========================================================
# Create Explanation DataFrame
# ==========================================================

xai_df = X_shap.copy()


# ----------------------------------------------------------
# Add Actual Label
# ----------------------------------------------------------

xai_df.loc[:, "Actual_Label"] = np.asarray(
    y_shap
)

# ----------------------------------------------------------
# Add Actual Status
# ----------------------------------------------------------

xai_df.loc[:, "Actual_Status"] = [
    class_names[int(label)]
    for label in y_shap
]

# ----------------------------------------------------------
# Add Prediction
# ----------------------------------------------------------

xai_df.loc[:, "Predicted_Label"] = (
    predicted_class
)

xai_df.loc[:, "Predicted_Status"] = (
    predicted_status
)

# ----------------------------------------------------------
# Add Confidence
# ----------------------------------------------------------

xai_df.loc[:, "Confidence_Score"] = np.round(
    confidence_scores,
    4
)

xai_df.loc[:, "Confidence_Percentage"] = np.round(
    confidence_scores * 100,
    2
)

# ----------------------------------------------------------
# Add Dominant Factor
# ----------------------------------------------------------

xai_df.loc[:, "Dominant_Factor"] = (
    dominant_factors
)

# ----------------------------------------------------------
# Add Secondary Factor
# ----------------------------------------------------------

xai_df.loc[:, "Secondary_Factor"] = (
    secondary_factors
)

# ----------------------------------------------------------
# Add SHAP Contributions
# ----------------------------------------------------------

xai_df.loc[:, "Dominant_Contribution"] = (
    dominant_contributions
)

xai_df.loc[:, "Secondary_Contribution"] = (
    secondary_contributions
)

# ==========================================================
# Generate Contribution Direction
# ==========================================================

dominant_direction = []


for value in dominant_contributions:

    if value > 0:

        dominant_direction.append(
            "Positive"
        )

    elif value < 0:

        dominant_direction.append(
            "Negative"
        )

    else:

        dominant_direction.append(
            "Neutral"
        )


xai_df["Dominant_Direction"] = (
    dominant_direction
)


# ==========================================================
# Display Summary
# ==========================================================

print()
print("=" * 60)
print("XAI Feature Contribution Summary")
print("=" * 60)


print()

print(
    "Dominant Factors:"
)


print(
    xai_df[
        "Dominant_Factor"
    ].value_counts()
)


print()

print(
    "Prediction Distribution:"
)


print(
    xai_df[
        "Predicted_Status"
    ].value_counts()
)


print()

print(
    "Confidence Statistics:"
)


print(
    xai_df[
        "Confidence_Score"
    ].describe()
)


print()

print(
    "Feature Contribution Generation Completed"
)


print()

print(
    "Next Step: Generate SHAP Explanation Text"
)

print("=" * 60)

# ==========================================================
# PART 4
# Generate Explanation + Recommendation
# Save XAI Results + SHAP Visualizations
# ==========================================================


print()
print("=" * 60)
print("Generating XAI Explanations")
print("=" * 60)

# ==========================================================
# Generate Human-Readable XAI Explanation
# ==========================================================

print()
print("=" * 60)
print("Generating XAI Explanations")
print("=" * 60)

explanations = []

for i in range(len(xai_df)):

    status = xai_df.iloc[i]["Predicted_Status"]

    dominant = xai_df.iloc[i]["Dominant_Factor"]

    secondary = xai_df.iloc[i]["Secondary_Factor"]

    confidence = float(
        xai_df.iloc[i]["Confidence_Score"]
    )

    dominant_value = float(
        xai_df.iloc[i]["Dominant_Contribution"]
    )

    secondary_value = float(
        xai_df.iloc[i]["Secondary_Contribution"]
    )

    if status == "Healthy":

        explanation = (
            f"The model predicted the cow as Healthy "
            f"with {confidence:.1%} confidence. "
            f"The most influential observed factor was "
            f"{dominant}, followed by {secondary}."
        )

    elif status == "At_Risk":

        explanation = (
            f"The model classified the cow as At Risk "
            f"with {confidence:.1%} confidence. "
            f"The dominant contributing factor was "
            f"{dominant}, followed by {secondary}. "
            f"These factors contributed to the model's "
            f"risk assessment."
        )

    else:

        explanation = (
            f"The model classified the cow as Diseased "
            f"with {confidence:.1%} confidence. "
            f"The strongest contributing factor was "
            f"{dominant}, followed by {secondary}. "
            f"The combination of these factors strongly "
            f"influenced the disease prediction."
        )

    explanations.append(explanation)


# ----------------------------------------------------------
# Save Explanation
# ----------------------------------------------------------

xai_df.loc[:, "XAI_Explanation"] = explanations

print()
print("Human-Readable Explanations Generated")

# ==========================================================
# SHAP Global Feature Importance
# ==========================================================

print()
print("=" * 60)
print("Generating SHAP Feature Importance")
print("=" * 60)


# ----------------------------------------------------------
# Combine absolute SHAP values from all classes
# ----------------------------------------------------------

mean_healthy = np.mean(
    np.abs(shap_healthy),
    axis=0
)

mean_at_risk = np.mean(
    np.abs(shap_at_risk),
    axis=0
)

mean_diseased = np.mean(
    np.abs(shap_diseased),
    axis=0
)


global_importance = (

    mean_healthy
    + mean_at_risk
    + mean_diseased

) / 3


# ==========================================================
# Create Importance DataFrame
# ==========================================================

shap_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Mean_SHAP_Importance":
        global_importance

})


shap_importance_df = (
    shap_importance_df
    .sort_values(
        by="Mean_SHAP_Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ----------------------------------------------------------
# Human-readable feature names
# ----------------------------------------------------------

shap_importance_df[
    "Feature_Display"
] = (

    shap_importance_df[
        "Feature"
    ].map(
        feature_display_names
    )

    .fillna(
        shap_importance_df[
            "Feature"
        ]
    )
)


# ==========================================================
# Save SHAP Importance
# ==========================================================

shap_importance_file = os.path.join(

    RESULT_DIR,

    "shap_feature_importance.csv"
)


shap_importance_df.to_csv(

    shap_importance_file,

    index=False
)


print()

print(
    "SHAP Feature Importance Saved:"
)

print(
    shap_importance_file
)


# ==========================================================
# SHAP Feature Importance Plot
# ==========================================================

print()
print(
    "Generating SHAP Feature Importance Plot..."
)


plot_df = (
    shap_importance_df
    .head(12)
    .sort_values(
        by="Mean_SHAP_Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(

    plot_df[
        "Feature_Display"
    ],

    plot_df[
        "Mean_SHAP_Importance"
    ]
)


plt.xlabel(
    "Mean Absolute SHAP Value"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Global SHAP Feature Importance"
)


plt.tight_layout()


shap_importance_plot = os.path.join(

    RESULT_DIR,

    "shap_feature_importance.png"
)


plt.savefig(

    shap_importance_plot,

    dpi=300
)


plt.close()


print(
    "SHAP Feature Importance Plot Saved"
)


# ==========================================================
# SHAP Summary Plot
# ==========================================================

print()
print(
    "Generating SHAP Summary Plot..."
)


# ----------------------------------------------------------
# Use predicted-class SHAP values
# ----------------------------------------------------------

predicted_shap_matrix = np.zeros_like(
    shap_healthy
)


for i in range(
    len(X_shap)
):

    predicted = int(
        predicted_class[i]
    )


    if predicted == 0:

        predicted_shap_matrix[i] = (
            shap_healthy[i]
        )


    elif predicted == 1:

        predicted_shap_matrix[i] = (
            shap_at_risk[i]
        )


    else:

        predicted_shap_matrix[i] = (
            shap_diseased[i]
        )


# ==========================================================
# Summary Plot
# ==========================================================

plt.figure(
    figsize=(10, 8)
)


shap.summary_plot(

    predicted_shap_matrix,

    X_shap,

    show=False,

    max_display=12
)


plt.tight_layout()


summary_plot_file = os.path.join(

    RESULT_DIR,

    "shap_summary.png"
)


plt.savefig(

    summary_plot_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "SHAP Summary Plot Saved"
)


# ==========================================================
# Save XAI Dataset
# ==========================================================

print()
print("=" * 60)
print("Saving XAI Dataset")
print("=" * 60)


xai_df.to_csv(

    OUTPUT_FILE,

    index=False
)


print()

print(
    "XAI Dataset Saved:"
)

print(
    OUTPUT_FILE
)


# ==========================================================
# Display Top SHAP Features
# ==========================================================

print()
print("=" * 60)
print("Top SHAP Features")
print("=" * 60)


for i, row in enumerate(

    shap_importance_df.head(10)
    .itertuples(index=False),

    start=1

):

    print(

        f"{i:02d}. "
        f"{row.Feature_Display:<25} "
        f"{row.Mean_SHAP_Importance:.6f}"

    )


# ==========================================================
# XAI Distribution
# ==========================================================

print()
print("=" * 60)
print("XAI Prediction Distribution")
print("=" * 60)


print()

print(
    xai_df[
        "Predicted_Status"
    ].value_counts()
)


# ==========================================================
# Final Statistics
# ==========================================================

print()
print("=" * 60)
print("XAI Confidence Statistics")
print("=" * 60)


print(
    xai_df[
        "Confidence_Score"
    ].describe()
)


# ==========================================================
# Final Output
# ==========================================================

print()
print("=" * 60)
print("SHAP XAI ENGINE COMPLETED SUCCESSFULLY")
print("=" * 60)


print()

print(
    "Total XAI Samples :",
    len(xai_df)
)


print()

print("Generated Files")
print("------------------------------------------")


print(
    "XAI Dataset"
)

print(
    "  generated/shap_explanations.csv"
)


print()

print(
    "SHAP Visualization"
)

print(
    "  results/shap_summary.png"
)

print(
    "  results/shap_feature_importance.png"
)


print()

print(
    "SHAP Feature Importance"
)

print(
    "  results/shap_feature_importance.csv"
)


print()

print(
    "XAI Engine is Ready"
)

print(
    "The model predictions can now be explained "
    "using SHAP-based feature contributions."
)


print("=" * 60)