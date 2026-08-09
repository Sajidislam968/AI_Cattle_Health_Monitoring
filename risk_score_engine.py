# ==========================================================
# risk_score_engine.py
# PART 1
#
# Literature Based Risk Score Calculation
# ==========================================================

import os
import numpy as np
import pandas as pd

# ==========================================================
# File Path
# ==========================================================

INPUT_FILE = "generated/disease_dataset.csv"
OUTPUT_FILE = "generated/risk_score_dataset.csv"

# ==========================================================
# Load Dataset
# ==========================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found."
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("="*60)
print("Disease Dataset Loaded Successfully")
print("="*60)
print(df.shape)
print()

# ==========================================================
# Literature Based Feature Weights
#
# (Based on your 14 papers)
# ==========================================================

WEIGHTS = {

    "Heart_Rate_BPM":0.18,

    "Body_Temperature_C":0.20,

    "Rumination_Min":0.15,

    "Eating_Hr":0.10,

    "Drinking_Min":0.05,

    "Walking_km":0.10,

    "Standing_Hr":0.05,

    "Lying_Hr":0.05,

    "Activity_Index":0.07,

    "THI":0.05

}

print("Total Weight =",sum(WEIGHTS.values()))
print()

# ==========================================================
# Normal Healthy Ranges
#
# (Collected from NotebookLM)
# ==========================================================

NORMAL = {

    "Heart_Rate_BPM":(48,84),

    "Body_Temperature_C":(38.5,39.5),

    "Rumination_Min":(400,520),

    "Eating_Hr":(4,6),

    "Drinking_Min":(8,20),

    "Walking_km":(3,8),

    "Standing_Hr":(6,12),

    "Lying_Hr":(6,14),

    "Activity_Index":(60,100),

    "THI":(0,72)

}

# ==========================================================
# Abnormality Score
#
# Returns value between 0 and 1
# ==========================================================

def abnormality_score(value, low, high):

    if low <= value <= high:
        return 0.0

    if value < low:

        score = (low - value) / low

    else:

        score = (value - high) / high

    return min(score,1.0)

# ==========================================================
# Risk Score Function
# ==========================================================

def calculate_risk_score(row):

    score = 0

    for feature, weight in WEIGHTS.items():

        low, high = NORMAL[feature]

        feature_score = abnormality_score(
            row[feature],
            low,
            high
        )

        score += feature_score * weight
        # ------------------------------------------------
        # Disease State Bonus
        # ------------------------------------------------
        if row["Disease_Label"] == 0:
            score += np.random.uniform(0.00, 0.08)

        if row["Disease_Label"] == 1:
            score += np.random.uniform(0.30, 0.60)

        elif row["Disease_Label"] == 2:
            score += np.random.uniform(0.60, 0.95)

        return min(score * 100, 100)
# ==========================================================
# PART 2
# Calculate Risk Score for Every Sample
# ==========================================================

risk_scores = []

print("Calculating Risk Scores...")

for _, row in df.iterrows():

    score = calculate_risk_score(row)

    # Keep between 0-100
    score = max(0, min(score, 100))

    risk_scores.append(round(score, 2))

df["Risk_Score"] = risk_scores

print("Risk Score Calculation Completed")
print()

# ==========================================================
# Assign Risk Level
# ==========================================================

risk_levels = []

for score in df["Risk_Score"]:

    if score < 25:
        risk_levels.append("Low")

    elif score < 50:
        risk_levels.append("Moderate")

    elif score < 75:
        risk_levels.append("High")

    else:
        risk_levels.append("Critical")

df["Risk_Level"] = risk_levels

print(df["Risk_Level"].value_counts())
print()
print(df["Risk_Score"].describe())
print()

print(df.groupby("Health_Status")["Risk_Score"].describe())

# ==========================================================
# Create Critical Alert
# ==========================================================

critical_alert = []

for score in df["Risk_Score"]:

    if score >= 75:
        critical_alert.append(1)
    else:
        critical_alert.append(0)

df["Critical_Alert"] = critical_alert

print(df["Critical_Alert"].value_counts())
print()

# ==========================================================
# Create Alert Reason
# ==========================================================

alert_reason = []

for _, row in df.iterrows():

    reasons = []

    if row["Heart_Rate_BPM"] > 84:
        reasons.append("High Heart Rate")

    if row["Heart_Rate_BPM"] < 48:
        reasons.append("Low Heart Rate")

    if row["Body_Temperature_C"] > 39.5:
        reasons.append("High Body Temperature")

    if row["Body_Temperature_C"] < 38.5:
        reasons.append("Low Body Temperature")

    if row["Rumination_Min"] < 400:
        reasons.append("Low Rumination")

    if row["Eating_Hr"] < 4:
        reasons.append("Low Eating")

    if row["Drinking_Min"] < 8:
        reasons.append("Low Drinking")

    if row["Walking_km"] < 3:
        reasons.append("Low Walking")

    if row["Activity_Index"] < 60:
        reasons.append("Low Activity")

    if row["THI"] >= 89:
        reasons.append("Severe Heat Stress")

    elif row["THI"] >= 79:
        reasons.append("Moderate Heat Stress")

    elif row["THI"] >= 72:
        reasons.append("Mild Heat Stress")

    if len(reasons) == 0:
        reasons.append("Healthy")

    alert_reason.append(", ".join(reasons))

df["Alert_Reason"] = alert_reason

print("Alert Reason Generated")
print()
# ==========================================================
# PART 3
# Recommendation Engine
# ==========================================================

recommendations = []

for _, row in df.iterrows():

    rec = []

    # ------------------------------------------------------
    # Heart Rate
    # ------------------------------------------------------
    if row["Heart_Rate_BPM"] > 84:
        rec.append("Check stress level and contact veterinarian")

    elif row["Heart_Rate_BPM"] < 48:
        rec.append("Check for weakness or cardiovascular problems")

    # ------------------------------------------------------
    # Body Temperature
    # ------------------------------------------------------
    if row["Body_Temperature_C"] > 39.5:
        rec.append("Possible fever. Immediate veterinary inspection recommended")

    elif row["Body_Temperature_C"] < 38.5:
        rec.append("Monitor for milk fever or metabolic disorder")

    # ------------------------------------------------------
    # Rumination
    # ------------------------------------------------------
    if row["Rumination_Min"] < 400:
        rec.append("Monitor digestive health and rumination activity")

    # ------------------------------------------------------
    # Eating
    # ------------------------------------------------------
    if row["Eating_Hr"] < 4:
        rec.append("Check feed quality and appetite")

    # ------------------------------------------------------
    # Drinking
    # ------------------------------------------------------
    if row["Drinking_Min"] < 8:
        rec.append("Ensure clean water availability and hydration")

    # ------------------------------------------------------
    # Walking
    # ------------------------------------------------------
    if row["Walking_km"] < 3:
        rec.append("Inspect for lameness or hoof injury")

    # ------------------------------------------------------
    # Standing
    # ------------------------------------------------------
    if row["Standing_Hr"] > 12:
        rec.append("Observe standing behavior for discomfort")

    # ------------------------------------------------------
    # Lying
    # ------------------------------------------------------
    if row["Lying_Hr"] > 14:
        rec.append("Excessive lying detected. Check for pain or illness")

    # ------------------------------------------------------
    # Activity
    # ------------------------------------------------------
    if row["Activity_Index"] < 60:
        rec.append("Low activity detected. Monitor closely")

    # ------------------------------------------------------
    # Heat Stress
    # ------------------------------------------------------
    if row["THI"] >= 89:
        rec.append("Emergency heat stress. Activate cooling system immediately")

    elif row["THI"] >= 79:
        rec.append("Provide shade, fans and additional drinking water")

    elif row["THI"] >= 72:
        rec.append("Monitor for mild heat stress")

    # ------------------------------------------------------
    # Healthy Animal
    # ------------------------------------------------------
    if len(rec) == 0:
        rec.append("No action required. Cow is healthy.")

    recommendations.append(" | ".join(rec))

df["Recommendation"] = recommendations

print("Recommendations Generated Successfully")
print()
# ==========================================================
# PART 4
# Save Final Dataset
# ==========================================================

# Arrange columns
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
    "Recommendation"
]

df = df[column_order]

# ==========================================================
# Save CSV
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("Risk Score Engine Completed Successfully")
print("=" * 60)

print(f"Total Samples : {len(df)}")
print(f"Output File   : {OUTPUT_FILE}")
print()

print("Risk Level Distribution")
print(df["Risk_Level"].value_counts())
print()

print("Disease Distribution")
print(df["Disease_Name"].value_counts())
print()

print("Critical Alerts")
print(df["Critical_Alert"].value_counts())
print()

print("Dataset Preview")
print(df.head())

print("=" * 60)
