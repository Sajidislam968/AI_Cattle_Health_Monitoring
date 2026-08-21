"""
============================================================
Real-Time Multimodal Inference Engine
============================================================

Pipeline

Video
   ↓
Vision Module
   ↓
Behavior Detection

+

Sensor AI Prediction

↓

Decision Fusion

↓

Final Output
"""

import os
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# Root Directory
# ==========================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("Real-Time Multimodal Inference Engine")
print("=" * 60)

# ==========================================================
# Folder Paths
# ==========================================================

VISION_OUTPUT = os.path.join(
    ROOT_DIR,
    "vision",
    "output",
    "multimodal_features.csv"
)

SENSOR_DATA = os.path.join(
    ROOT_DIR,
    "generated",
    "xai_dataset.csv"
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "xgboost_model.pkl"
)

DECISION_OUTPUT = os.path.join(
    ROOT_DIR,
    "output",
    "decision_fusion_results.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nChecking required files...\n")

required = [
    VISION_OUTPUT,
    SENSOR_DATA,
    MODEL_PATH,
    DECISION_OUTPUT
]

missing = []

for file in required:

    if not os.path.exists(file):
        missing.append(file)

if len(missing):

    print("Missing Files\n")

    for f in missing:
        print(f)

    raise FileNotFoundError()

print("All required files found.")

print("\nLoading resources...\n")

vision_df = pd.read_csv(VISION_OUTPUT)

sensor_df = pd.read_csv(
    SENSOR_DATA,
    low_memory=False
)

decision_df = pd.read_csv(
    DECISION_OUTPUT
)

model = joblib.load(MODEL_PATH)

print("Vision :", vision_df.shape)
print("Sensor :", sensor_df.shape)
print("Decision :", decision_df.shape)

print("\nXGBoost model loaded successfully.")

# ==========================================================
# Prepare Latest Vision State
# ==========================================================

print("\nPreparing latest vision state...\n")

vision_df["Cow_ID"] = (
    vision_df["Cow_ID"]
    .astype(str)
    .str.strip()
    .str.lower()
)

vision_latest = (
    vision_df
    .sort_values("Frame")
    .groupby("Cow_ID", as_index=False)
    .last()
)

print("Latest Vision Shape :", vision_latest.shape)

print("\nVision Preview\n")

print(
    vision_latest[
        [
            "Cow_ID",
            "Frame",
            "Behavior_Label_behavior"
        ]
    ].head(10)
)
# ==========================================================
# Prepare Latest Sensor State
# ==========================================================

print("\nPreparing latest sensor state...\n")

sensor_df["Cow_ID"] = (
    sensor_df["Cow_ID"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sensor_df["Cow_ID"] = sensor_df["Cow_ID"].str.replace(
    r"cow_0*([0-9]+)",
    r"cow_\1",
    regex=True
)

sensor_df = sensor_df.sort_values(
    by=[
        "Cow_ID",
        "Day",
        "Hour"
    ]
)

sensor_latest = (
    sensor_df
    .groupby("Cow_ID", as_index=False)
    .last()
)

print("Latest Sensor Shape :", sensor_latest.shape)

print("\nSensor Preview\n")

print(
    sensor_latest[
        [
            "Cow_ID",
            "Heart_Rate_BPM",
            "Body_Temperature_C",
            "Rumination_Min",
            "Activity_Index",
            "Health_Status"
        ]
    ].head(10)
)
# ==========================================================
# Merge Current Cow State
# ==========================================================

print("\nBuilding current multimodal state...\n")

current_state = pd.merge(
    vision_latest,
    sensor_latest,
    on="Cow_ID",
    how="inner"
)

print("Current State Shape :", current_state.shape)

print("\nCurrent State Preview\n")

print(
    current_state[
        [
            "Cow_ID",
            "Behavior_Label_behavior",
            "Heart_Rate_BPM",
            "Body_Temperature_C",
            "Health_Status"
        ]
    ].head(15)
)
print("\nNumber of cows from Vision :", len(vision_latest))
print("Number of cows from Sensor :", len(sensor_latest))
print("Matched cows :", len(current_state))

# ==========================================================
# Features for XGBoost Prediction
# ==========================================================

print("\nPreparing features for XGBoost...\n")

FEATURE_COLUMNS = [

    "Heart_Rate_BPM",

    "Body_Temperature_C",

    "Rumination_Min",

    "Eating_Hr",

    "Drinking_Min",

    "Walking_km",

    "Standing_Hr",

    "Lying_Hr",

    "Activity_Index",

    "Ambient_Temperature_C",

    "Relative_Humidity",

    "THI"

]

missing = []

for col in FEATURE_COLUMNS:

    if col not in current_state.columns:

        missing.append(col)

if len(missing):

    print("Missing Features")

    print(missing)

    raise ValueError()

X = current_state[FEATURE_COLUMNS].copy()

print("Feature Matrix Shape")

print(X.shape)

print("\nPreview\n")

print(X.head())

# ==========================================================
# Predict Health Status
# ==========================================================

print("\nRunning XGBoost prediction...\n")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = probabilities.max(axis=1)

current_state["Predicted_Label"] = predictions

current_state["Prediction_Confidence"] = confidence

# ==========================================================
# Decode XGBoost Prediction
# ==========================================================

print("\nDecoding predictions...\n")

LABEL_MAP = {

    0: "Healthy",

    1: "At Risk",

    2: "Diseased"

}

current_state["Predicted_Status"] = (
    current_state["Predicted_Label"]
    .map(LABEL_MAP)
)

current_state["Prediction_Confidence"] = (
    current_state["Prediction_Confidence"] * 100
).round(2)

print("Prediction decoding completed.")

# ==========================================================
# AI Prediction Summary
# ==========================================================

print("\nCreating AI prediction summary...\n")

current_state["AI_Summary"] = (
    "Predicted "
    + current_state["Predicted_Status"]
    + " ("
    + current_state["Prediction_Confidence"].astype(str)
    + "% confidence)"
)
# ==========================================================
# Prediction Statistics
# ==========================================================

print("\nPrediction Distribution\n")

print(
    current_state["Predicted_Status"]
    .value_counts()
)

print("\nAverage Confidence")

print(
    round(
        current_state["Prediction_Confidence"].mean(),
        2
    ),
    "%"
)
print("\nPrediction Preview\n")

print(

current_state[
[
    "Cow_ID",

    "Behavior_Label_behavior",

    "Predicted_Status",

    #"Prediction_Confidence",

    "AI_Summary"
]

].head(20)

)
# ==========================================================
# Decision Fusion
# ==========================================================

print("\nRunning Decision Fusion...\n")

current_state["Final_Status"] = ""
current_state["Alert_Level"] = ""
current_state["Decision_Source"] = ""
current_state["Final_Recommendation"] = ""
for idx, row in current_state.iterrows():

    behavior = str(
        row["Behavior_Label_behavior"]
    ).lower()

    status = str(
        row["Predicted_Status"]
    ).lower()

    confidence = float(
        row["Prediction_Confidence"]
    )

    if status == "diseased":

        current_state.loc[idx, "Final_Status"] = "Diseased"

        current_state.loc[idx, "Alert_Level"] = "Critical"

        current_state.loc[idx, "Decision_Source"] = "Sensor AI"

        current_state.loc[idx, "Final_Recommendation"] = (
            "Immediate veterinary examination required."
        )

        continue
    if status == "at risk":

        current_state.loc[idx, "Final_Status"] = "At Risk"

        current_state.loc[idx, "Alert_Level"] = "Medium"

        current_state.loc[idx, "Decision_Source"] = "Sensor AI"

        current_state.loc[idx, "Final_Recommendation"] = (
            "Monitor closely and repeat health assessment."
        )

        continue
    if status == "healthy":

        current_state.loc[idx, "Final_Status"] = "Healthy"

        current_state.loc[idx, "Alert_Level"] = "Low"

        current_state.loc[idx, "Decision_Source"] = "Vision + Sensor"

        current_state.loc[idx, "Final_Recommendation"] = (
            "Continue routine monitoring."
        )

    abnormal_behavior = [
        "falling",
        "limping",
        "collapsed",
        "abnormal"
    ]

    if behavior in abnormal_behavior:

        current_state.loc[idx, "Final_Status"] = (
            "Observation Needed"
        )

        current_state.loc[idx, "Alert_Level"] = "Medium"

        current_state.loc[idx, "Decision_Source"] = (
            "Vision AI"
        )

        current_state.loc[idx, "Final_Recommendation"] = (
            "Abnormal behaviour detected. Observe animal."
        )

print("\nDecision Fusion Completed.\n")

print(

current_state[
[ 
"Cow_ID",

"Behavior_Label_behavior",

"Predicted_Status",

"Final_Status",

"Alert_Level",

"Decision_Source"

]

].head(20)

)

# ==========================================================
# Dynamic Recommendation Engine
# ==========================================================

print("\nGenerating Dynamic Recommendations...\n")

current_state["Dynamic_Recommendation"] = ""

for idx, row in current_state.iterrows():

    status = row["Final_Status"]

    behavior = str(
        row["Behavior_Label_behavior"]
    ).lower()

    recommendation = ""

    # ---------------------------------------
    # Healthy
    # ---------------------------------------

    if status == "Healthy":

        recommendation = (
            "Cow appears healthy. Continue routine monitoring."
        )

    # ---------------------------------------
    # At Risk
    # ---------------------------------------

    elif status == "At Risk":

        recommendation = (
            "Monitor the cow closely over the next few hours."
        )

        if behavior == "lying":

            recommendation += (
                " Extended lying behaviour observed."
            )

        elif behavior == "walking":

            recommendation += (
                " Observe walking pattern carefully."
            )

        elif behavior == "rumination":

            recommendation += (
                " Check rumination consistency."
            )

    # ---------------------------------------
    # Diseased
    # ---------------------------------------

    elif status == "Diseased":

        recommendation = (
            "Veterinary examination is recommended immediately."
        )

        if behavior == "lying":

            recommendation += (
                " Cow is inactive."
            )

        elif behavior == "walking":

            recommendation += (
                " Reduced mobility may indicate illness."
            )

        elif behavior == "rumination":

            recommendation += (
                " Verify feed intake and rumination."
            )

    # ---------------------------------------
    # Emergency
    # ---------------------------------------

    elif status == "Emergency":

        recommendation = (
            "Emergency detected. Isolate the cow and contact a veterinarian immediately."
        )

    # ---------------------------------------
    # Observation Needed
    # ---------------------------------------

    else:

        recommendation = (
            "Observe behaviour for several minutes before making a clinical decision."
        )

    current_state.loc[
        idx,
        "Dynamic_Recommendation"
    ] = recommendation

# ==========================================================
# Recommendation Priority
# ==========================================================

priority = {

    "Healthy": "Low",

    "At Risk": "Medium",

    "Diseased": "High",

    "Emergency": "Critical",

    "Observation Needed": "Medium"

}

current_state["Recommendation_Priority"] = (
    current_state["Final_Status"]
    .map(priority)
)

print("\nRecommendation Preview\n")

print(

current_state[
[
    "Cow_ID",

    "Final_Status",

    "Behavior_Label_behavior",

    "Recommendation_Priority",

    "Dynamic_Recommendation"

]

].head(20)

)

# ==========================================================
# Prepare Dashboard Dataset
# ==========================================================

print("\nPreparing Dashboard Dataset...\n")

dashboard_df = current_state[
[
    "Cow_ID",
    "Frame",
    "Behavior_Label_behavior",
    "Predicted_Status",
    "Prediction_Confidence",
    "Final_Status",
    "Alert_Level",
    "Decision_Source",
    "Recommendation_Priority",
    "Dynamic_Recommendation"
]
].copy()

dashboard_df = dashboard_df.sort_values(
    by="Cow_ID"
).reset_index(drop=True)

dashboard_dir = os.path.join(
    ROOT_DIR,
    "results"
)

os.makedirs(
    dashboard_dir,
    exist_ok=True
)

dashboard_file = os.path.join(
    dashboard_dir,
    "dashboard_dataset.csv"
)

dashboard_df.to_csv(
    dashboard_file,
    index=False
)

print("\nDashboard Dataset Saved Successfully")

print(dashboard_file)

print("\nShape")

print(dashboard_df.shape)

print("\nPreview\n")

print(
    dashboard_df.head(20)
)