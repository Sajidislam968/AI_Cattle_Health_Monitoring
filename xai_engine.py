# ==========================================================
# xai_engine.py
#
# Phase-4
# Explainable AI (XAI) Engine
#
# Input:
# generated/risk_score_dataset.csv
#
# Output:
# generated/xai_dataset.csv
#
# ==========================================================

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------
# File Paths
# ----------------------------------------------------------

INPUT_FILE = "generated/risk_score_dataset.csv"
OUTPUT_FILE = "generated/xai_dataset.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("=" * 60)
print("Risk Score Dataset Loaded Successfully")
print("=" * 60)
print(df.shape)

# ----------------------------------------------------------
# Feature Importance
# (Literature Based)
# ----------------------------------------------------------

FEATURE_IMPORTANCE = {

    "Heart_Rate_BPM": 0.18,
    "Body_Temperature_C": 0.22,
    "Rumination_Min": 0.18,
    "Activity_Index": 0.15,
    "Walking_km": 0.10,
    "Eating_Hr": 0.07,
    "Standing_Hr": 0.04,
    "Lying_Hr": 0.03,
    "THI": 0.03

}

# ----------------------------------------------------------
# Normal Reference Values
# ----------------------------------------------------------

NORMAL = {

    "Heart_Rate_BPM": (48,84),
    "Body_Temperature_C": (38.5,39.5),
    "Rumination_Min": (400,520),
    "Eating_Hr": (4,6),
    "Walking_km": (4,8),
    "Standing_Hr": (6,12),
    "Lying_Hr": (8,14),
    "Activity_Index": (60,100),
    "THI": (0,72)

}

# ----------------------------------------------------------
# Calculate Feature Deviations
# ----------------------------------------------------------

def feature_deviation(row):

    scores = {}

    # Heart Rate

    hr = row["Heart_Rate_BPM"]

    if hr < NORMAL["Heart_Rate_BPM"][0]:
        scores["Heart_Rate_BPM"] = NORMAL["Heart_Rate_BPM"][0] - hr

    elif hr > NORMAL["Heart_Rate_BPM"][1]:
        scores["Heart_Rate_BPM"] = hr - NORMAL["Heart_Rate_BPM"][1]

    else:
        scores["Heart_Rate_BPM"] = 0

    # Body Temperature

    bt = row["Body_Temperature_C"]

    if bt < NORMAL["Body_Temperature_C"][0]:
        scores["Body_Temperature_C"] = NORMAL["Body_Temperature_C"][0] - bt

    elif bt > NORMAL["Body_Temperature_C"][1]:
        scores["Body_Temperature_C"] = bt - NORMAL["Body_Temperature_C"][1]

    else:
        scores["Body_Temperature_C"] = 0

    # Rumination

    rum = row["Rumination_Min"]

    if rum < NORMAL["Rumination_Min"][0]:
        scores["Rumination_Min"] = NORMAL["Rumination_Min"][0] - rum

    else:
        scores["Rumination_Min"] = 0

    # Eating

    eat = row["Eating_Hr"]

    if eat < NORMAL["Eating_Hr"][0]:
        scores["Eating_Hr"] = NORMAL["Eating_Hr"][0] - eat

    else:
        scores["Eating_Hr"] = 0

        # Walking

    walk = row["Walking_km"]

    if walk < NORMAL["Walking_km"][0]:
        scores["Walking_km"] = NORMAL["Walking_km"][0] - walk

    else:
        scores["Walking_km"] = 0

    # Standing

    standing = row["Standing_Hr"]

    if standing > NORMAL["Standing_Hr"][1]:
        scores["Standing_Hr"] = standing - NORMAL["Standing_Hr"][1]

    else:
        scores["Standing_Hr"] = 0

    # Lying

    lying = row["Lying_Hr"]

    if lying < NORMAL["Lying_Hr"][0]:
        scores["Lying_Hr"] = NORMAL["Lying_Hr"][0] - lying

    elif lying > NORMAL["Lying_Hr"][1]:
        scores["Lying_Hr"] = lying - NORMAL["Lying_Hr"][1]

    else:
        scores["Lying_Hr"] = 0

    # Activity

    activity = row["Activity_Index"]

    if activity < NORMAL["Activity_Index"][0]:
        scores["Activity_Index"] = NORMAL["Activity_Index"][0] - activity

    else:
        scores["Activity_Index"] = 0

    # THI

    thi = row["THI"]

    if thi > NORMAL["THI"][1]:
        scores["THI"] = thi - NORMAL["THI"][1]

    else:
        scores["THI"] = 0

    return scores

# ----------------------------------------------------------
# Dominant & Secondary Feature
# ----------------------------------------------------------

def get_dominant_features(scores):

    weighted_scores = {}

    for feature in scores:

        weighted_scores[feature] = (
            scores[feature] *
            FEATURE_IMPORTANCE[feature]
        )

    sorted_features = sorted(
        weighted_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    dominant = sorted_features[0][0]
    secondary = sorted_features[1][0]

    contribution = FEATURE_IMPORTANCE[dominant] * 100

    return dominant, secondary, contribution

# ----------------------------------------------------------
# Human Readable Feature Name
# ----------------------------------------------------------

feature_name = {

    "Heart_Rate_BPM": "Heart Rate",

    "Body_Temperature_C": "Body Temperature",

    "Rumination_Min": "Rumination",

    "Eating_Hr": "Feeding",

    "Walking_km": "Walking Activity",

    "Standing_Hr": "Standing Duration",

    "Lying_Hr": "Lying Duration",

    "Activity_Index": "Activity",

    "THI": "Heat Stress"

}

# ----------------------------------------------------------
# Generate XAI Explanation
# ----------------------------------------------------------

def generate_explanation(row, dominant, secondary):

    disease = row["Disease_Name"]
    status = row["Health_Status"]

    # Healthy

    if status == "Healthy":

        return (
            "All physiological, behavioral and environmental "
            "parameters remain within their expected operating "
            "ranges. No significant abnormalities were detected."
        )

    # BRD

    if disease == "BRD":

        return (
            f"The primary contributor is {feature_name[dominant]}, "
            f"supported by changes in {feature_name[secondary]}. "
            "Reduced walking activity together with decreased "
            "overall activity and elevated body temperature "
            "indicate possible bovine respiratory disease."
        )

    # Mastitis

    elif disease == "Mastitis":

        return (
            f"The primary contributor is {feature_name[dominant]}, "
            f"supported by {feature_name[secondary]}. "
            "Elevated body temperature together with reduced "
            "rumination suggests inflammatory changes "
            "consistent with mastitis."
        )

    # Lameness

    elif disease == "Lameness":

        return (
            f"The primary contributor is {feature_name[dominant]}, "
            f"supported by {feature_name[secondary]}. "
            "Reduced walking distance together with prolonged "
            "standing duration indicates possible locomotion "
            "impairment or lameness."
        )

    # Heat Stress

    elif disease == "Heat_Stress":

        return (
            f"The primary contributor is {feature_name[dominant]}, "
            f"supported by {feature_name[secondary]}. "
            "High Temperature-Humidity Index (THI) together with "
            "behavioral changes indicates environmental heat stress."
        )

    # Default

    return (
        f"Abnormal changes were detected in "
        f"{feature_name[dominant]} and "
        f"{feature_name[secondary]}."
    )


# ----------------------------------------------------------
# Recommendation Generator (XAI Driven)
# ----------------------------------------------------------

def generate_recommendation(row, dominant):

    status = row["Health_Status"]
    disease = row["Disease_Name"]

    if status == "Healthy":

        return (
            "Continue routine monitoring. "
            "No intervention is currently required."
        )

    if disease == "Heat_Stress":

        return (
            "Increase clean water availability, activate cooling "
            "fans, provide shade and continuously monitor THI."
        )

    elif disease == "BRD":

        return (
            "Inspect respiratory condition, isolate the animal "
            "if necessary and consult a veterinarian."
        )

    elif disease == "Mastitis":

        return (
            "Inspect the udder, monitor body temperature and "
            "seek veterinary examination."
        )

    elif disease == "Lameness":

        return (
            "Inspect hooves, reduce walking load and keep the "
            "animal under close observation."
        )

    # Feature-based fallback

    if dominant == "Heart_Rate_BPM":

        return "Monitor heart rate continuously."

    elif dominant == "Body_Temperature_C":

        return "Monitor body temperature frequently."

    elif dominant == "Rumination_Min":

        return "Observe rumination and digestive health."

    elif dominant == "Eating_Hr":

        return "Monitor feeding behaviour."

    elif dominant == "Walking_km":

        return "Inspect locomotion and walking activity."

    elif dominant == "THI":

        return "Improve barn cooling and ventilation."

    return "Continue close monitoring."


# ----------------------------------------------------------
# Confidence Score
# ----------------------------------------------------------

def confidence_score(row):

    if row["Health_Status"] == "Healthy":

        return round(np.random.uniform(0.95, 1.00), 2)

    elif row["Health_Status"] == "At_Risk":

        return round(np.random.uniform(0.75, 0.90), 2)

    else:

        return round(np.random.uniform(0.90, 0.99), 2)

# ----------------------------------------------------------
# Generate XAI Columns
# ----------------------------------------------------------

print()
print("Generating XAI Explanations...")

dominant_list = []
secondary_list = []
contribution_list = []
explanation_list = []
recommendation_list = []
confidence_list = []

for _, row in df.iterrows():

    scores = feature_deviation(row)

    dominant, secondary, contribution = get_dominant_features(scores)

    dominant_list.append(feature_name[dominant])

    secondary_list.append(feature_name[secondary])

    contribution_list.append(f"{feature_name[dominant]} ({contribution:.0f}%)")

    explanation_list.append(
        generate_explanation(
            row,
            dominant,
            secondary
        )
    )

    recommendation_list.append(
        generate_recommendation(
            row,
            dominant
        )
    )

    confidence_list.append(
        confidence_score(row)
    )

# ----------------------------------------------------------
# Add New Columns
# ----------------------------------------------------------

df["Dominant_Factor"] = dominant_list
df["Secondary_Factor"] = secondary_list
df["Feature_Contribution"] = contribution_list
df["XAI_Explanation"] = explanation_list
df["Recommendation"] = recommendation_list
df["Confidence_Score"] = confidence_list

# ----------------------------------------------------------
# Arrange Columns
# ----------------------------------------------------------

column_order = [

    "Sample_ID",
    "Cow_ID",
    "Day",
    "Hour",

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
    "THI",

    "Health_Status",
    "Disease_Name",
    "Disease_Label",

    "Risk_Score",
    "Risk_Level",

    "Critical_Alert",
    "Alert_Reason",

    "Dominant_Factor",
    "Secondary_Factor",
    "Feature_Contribution",

    "XAI_Explanation",

    "Recommendation",

    "Confidence_Score"

]

df = df[column_order]

# ----------------------------------------------------------
# Save Dataset
# ----------------------------------------------------------

os.makedirs("generated", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print()
print("=" * 60)
print("XAI Engine Completed Successfully")
print("=" * 60)

print("Total Samples :", len(df))
print("Output File   :", OUTPUT_FILE)

print()

print("Health Status Distribution")
print(df["Health_Status"].value_counts())

print()

print("Risk Level Distribution")
print(df["Risk_Level"].value_counts())

print()

print("Dominant Factors")
print(df["Dominant_Factor"].value_counts())

print()

print("Confidence Score Statistics")
print(df["Confidence_Score"].describe())

print()

print("Dataset Preview")
print(df.head())

print("=" * 60)
