"""
===========================================================
Decision Fusion Engine
AI-Powered Multi-Modal Cattle Health Monitoring System
===========================================================

This module combines

1. Vision Module Output
2. Sensor AI Prediction
3. XAI Explanation

and produces

Final Health Status
Final Confidence
Final Decision
Recommendation Trigger
===========================================================
"""

import os
import pandas as pd
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("Decision Fusion Engine")
print("=" * 60)

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

VISION_FILE = os.path.join(
    BASE_DIR,
    "vision",
    "output",
    "multimodal_features.csv"
)

SENSOR_FILE = os.path.join(
    BASE_DIR,
    "generated",
    "xai_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "decision_fusion_output.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# Check Files
# ----------------------------------------------------

print("\nChecking required files...\n")

required_files = [
    VISION_FILE,
    SENSOR_FILE
]

missing = []

for file in required_files:

    if not os.path.exists(file):
        missing.append(file)

if len(missing):

    print("Missing Files:\n")

    for f in missing:
        print(f)

    raise FileNotFoundError()

print("All required files found.")

# ----------------------------------------------------
# Load Datasets
# ----------------------------------------------------

print("\nLoading datasets...\n")

vision_df = pd.read_csv(VISION_FILE)
sensor_df = pd.read_csv(
    SENSOR_FILE,
    low_memory=False
)

print(f"Vision Dataset : {vision_df.shape}")
print(f"Sensor Dataset : {sensor_df.shape}")

# ----------------------------------------------------
# Required Columns
# ----------------------------------------------------

vision_required = [
    "Cow_ID",
    "Behavior_Label_behavior"
]

sensor_required = [
    "Cow_ID",
    "Health_Status",
    "Confidence_Score",
    "Risk_Score",
    "Recommendation",
    "XAI_Explanation",
    "Dominant_Factor"
]

print("\nChecking required columns...")

for col in vision_required:

    if col not in vision_df.columns:
        raise ValueError(f"Missing Vision Column : {col}")

for col in sensor_required:

    if col not in sensor_df.columns:
        raise ValueError(f"Missing Sensor Column : {col}")

print("All required columns found.")

# ----------------------------------------------------
# Standardize Cow_ID
# ----------------------------------------------------

print("\nFormatting Cow_ID...")

vision_df["Cow_ID"] = (
    vision_df["Cow_ID"]
    .astype(str)
    .str.strip()
    .str.lower()
)

sensor_df["Cow_ID"] = (
    sensor_df["Cow_ID"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("cow_", "cow_", regex=False)
)

# যদি sensor dataset-এ COW_001 থাকে তাহলে cow_1 বানাবে
sensor_df["Cow_ID"] = sensor_df["Cow_ID"].str.replace(
    r"cow_0*([0-9]+)",
    r"cow_\1",
    regex=True
)

print("\nUnique Cow IDs")

print("Vision :", sorted(vision_df["Cow_ID"].unique())[:10])
print("Sensor :", sorted(sensor_df["Cow_ID"].unique())[:10])

# ----------------------------------------------------
# Preview
# ----------------------------------------------------

print("\nVision Preview")
print(
    vision_df[
        ["Cow_ID", "Behavior_Label_behavior"]
    ].head()
)

print("\nSensor Preview")
print(
    sensor_df[
        [
            "Cow_ID",
            "Health_Status",
            "Confidence_Score",
            "Risk_Score"
        ]
    ].head()
)

# ----------------------------------------------------
# Prepare Latest Sensor State
# ----------------------------------------------------

print("\nPreparing latest sensor state...\n")

# Sort by time
sensor_df = sensor_df.sort_values(
    by=["Cow_ID", "Day", "Hour"]
)

# Keep latest record of every cow
latest_sensor_df = (
    sensor_df
    .groupby("Cow_ID", as_index=False)
    .last()
)

print("Latest Sensor Dataset")
print(latest_sensor_df.shape)

print("\nPreview\n")

print(
    latest_sensor_df[
        [
            "Cow_ID",
            "Health_Status",
            "Risk_Score",
            "Confidence_Score",
            "Recommendation"
        ]
    ].head(10)
)

# ----------------------------------------------------
# Prepare Current Vision State
# ----------------------------------------------------

print("\nPreparing latest vision state...\n")

vision_latest = (
    vision_df
    .sort_values("Frame")
    .groupby("Cow_ID", as_index=False)
    .last()
)

print("Latest Vision Dataset")

print(vision_latest.shape)

print("\nPreview\n")

print(
    vision_latest[
        [
            "Cow_ID",
            "Frame",
            "Behavior_Label_behavior"
        ]
    ].head(10)
)

# ----------------------------------------------------
# Merge Vision + Sensor
# ----------------------------------------------------

print("\nMerging Vision and Sensor information...\n")

fusion_df = pd.merge(
    vision_latest,
    latest_sensor_df,
    on="Cow_ID",
    how="inner"
)

print("Fusion Dataset Shape :")
print(fusion_df.shape)

print("\nFusion Preview\n")

print(
    fusion_df[
        [
            "Cow_ID",
            "Behavior_Label_behavior",
            "Health_Status",
            "Risk_Score",
            "Confidence_Score"
        ]
    ].head(10)
)

# ----------------------------------------------------
# Keep Only Required Columns
# ----------------------------------------------------

fusion_df = fusion_df[
[
    "Cow_ID",
    "Frame",

    "Behavior_Label_behavior",
    #"Vision_Flag",

    "Health_Status",

    "Risk_Score",
    "Confidence_Score",

    "Recommendation",
    "XAI_Explanation",

    "Dominant_Factor",
    "Secondary_Factor",

    "Critical_Alert",

    #"Final_Status",
    #"Alert_Level",
    #"Decision_Source",
    #"Final_Recommendation"
]
]

print("\nSelected Columns")

print(fusion_df.columns.tolist())

print("\nCurrent Shape")

print(fusion_df.shape)

# ----------------------------------------------------
# Decision Fusion Rules
# ----------------------------------------------------

print("\nRunning Decision Fusion...\n")

fusion_df["Vision_Flag"] = "Normal"

# ----------------------------------------------------
# Vision Behaviour Analysis
# ----------------------------------------------------

NORMAL_BEHAVIOR = [
    "walking",
    "standing",
    "rumination",
    "eating"
]

ABNORMAL_BEHAVIOR = [
    "lying"
]

for idx, row in fusion_df.iterrows():

    behavior = str(row["Behavior_Label_behavior"]).lower().strip()

    if behavior in ABNORMAL_BEHAVIOR:
        fusion_df.loc[idx, "Vision_Flag"] = "Suspicious"

    else:
        fusion_df.loc[idx, "Vision_Flag"] = "Normal"

# ----------------------------------------------------
# Final Decision
# ----------------------------------------------------

fusion_df["Final_Status"] = ""
fusion_df["Alert_Level"] = ""
fusion_df["Decision_Source"] = ""
fusion_df["Final_Recommendation"] = ""

for idx, row in fusion_df.iterrows():

    health = str(row["Health_Status"]).lower()

    behavior_flag = row["Vision_Flag"]

    risk = float(row["Risk_Score"])

    confidence = float(row["Confidence_Score"])

    alert = int(row["Critical_Alert"])

    recommendation = row["Recommendation"]

    explanation = row["XAI_Explanation"]

    # ----------------------------------
    # Rule 1
    # Critical Alert
    # ----------------------------------

    if alert == 1:

        fusion_df.loc[idx, "Final_Status"] = "Emergency"

        fusion_df.loc[idx, "Alert_Level"] = "Critical"

        fusion_df.loc[idx, "Decision_Source"] = "Sensor AI"

        fusion_df.loc[idx, "Final_Recommendation"] = recommendation

        continue

    # ----------------------------------
    # Rule 2
    # Diseased from Sensor
    # ----------------------------------

    if health == "diseased":

        fusion_df.loc[idx, "Final_Status"] = "Diseased"

        fusion_df.loc[idx, "Alert_Level"] = "High"

        fusion_df.loc[idx, "Decision_Source"] = "Sensor AI"

        fusion_df.loc[idx, "Final_Recommendation"] = recommendation

        continue

    # ----------------------------------
    # Rule 3
    # At Risk
    # ----------------------------------

    if health == "at risk":

        if behavior_flag == "Suspicious":

            fusion_df.loc[idx, "Final_Status"] = "At Risk"

            fusion_df.loc[idx, "Alert_Level"] = "High"

            fusion_df.loc[idx, "Decision_Source"] = "Vision + Sensor"

        else:

            fusion_df.loc[idx, "Final_Status"] = "At Risk"

            fusion_df.loc[idx, "Alert_Level"] = "Medium"

            fusion_df.loc[idx, "Decision_Source"] = "Sensor AI"

        fusion_df.loc[idx, "Final_Recommendation"] = recommendation

        continue

    # ----------------------------------
    # Rule 4
    # Healthy
    # ----------------------------------

    if health == "healthy":

        if behavior_flag == "Suspicious":

            fusion_df.loc[idx, "Final_Status"] = "Observation Needed"

            fusion_df.loc[idx, "Alert_Level"] = "Medium"

            fusion_df.loc[idx, "Decision_Source"] = "Vision AI"

            fusion_df.loc[idx, "Final_Recommendation"] = (
                "Cow appears healthy physiologically but behavior is abnormal. Continue monitoring."
            )

        else:

            fusion_df.loc[idx, "Final_Status"] = "Healthy"

            fusion_df.loc[idx, "Alert_Level"] = "Low"

            fusion_df.loc[idx, "Decision_Source"] = "Vision + Sensor"

            fusion_df.loc[idx, "Final_Recommendation"] = recommendation
# ----------------------------------------------------
# Save Final Decision Dataset
# ----------------------------------------------------

print("\nSaving Decision Fusion Dataset...\n")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "decision_fusion_results.csv"
)

fusion_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Decision Fusion Completed Successfully!")

print("\nSaved File")

print(OUTPUT_FILE)

# ----------------------------------------------------
# Dataset Summary
# ----------------------------------------------------

print("\nDataset Summary")
print("-" * 50)

print("Rows :", len(fusion_df))
print("Columns :", len(fusion_df.columns))

print("\nColumns")

print(fusion_df.columns.tolist())

# ----------------------------------------------------
# Final Status Distribution
# ----------------------------------------------------

print("\nFinal Status Distribution")

print(
    fusion_df["Final_Status"]
    .value_counts()
)
# ----------------------------------------------------
# Alert Level Distribution
# ----------------------------------------------------

print("\nAlert Level Distribution")

print(
    fusion_df["Alert_Level"]
    .value_counts()
)

# ----------------------------------------------------
# Sample Results
# ----------------------------------------------------

print("\nSample Decision Results\n")

print(

    fusion_df[
        [

            "Cow_ID",

            "Behavior_Label_behavior",

            "Health_Status",

            "Risk_Score",

            "Confidence_Score",

            "Final_Status",

            "Alert_Level",

            "Decision_Source"

        ]

    ].head(20)

)
