"""
============================================================
Decision Fusion Engine
============================================================

Fusion of

1. Vision Behavior
2. Sensor Risk Score

Output:
generated/final_prediction.csv
"""

import pandas as pd

print("=" * 60)
print("Decision Fusion Engine")
print("=" * 60)

# ----------------------------------------------------
# Load Files
# ----------------------------------------------------

behavior = pd.read_csv("vision/output/behavior_summary.csv")
risk = pd.read_csv("generated/risk_scores.csv")

# ----------------------------------------------------
# Merge
# ----------------------------------------------------

df = pd.merge(
    behavior,
    risk,
    left_on="Cow_ID",
    right_on="Cow_ID",
    how="inner"
)

# ----------------------------------------------------
# Decision Rules
# ----------------------------------------------------

def final_status(row):

    risk = row["Risk_Level"]
    behavior = row["Dominant_Behavior"]

    if risk == "Diseased":
        return "Diseased"

    if risk == "At Risk":

        if behavior in [
            "lying",
            "rumination"
        ]:
            return "Diseased"

        return "At Risk"

    # Healthy Risk

    if behavior == "lying":
        return "Observation"

    return "Healthy"

# ----------------------------------------------------

df["Final_Status"] = df.apply(
    final_status,
    axis=1
)

# ----------------------------------------------------
# Confidence
# ----------------------------------------------------

confidence = []

for _, row in df.iterrows():

    score = row["Risk_Score"]

    if score >= 80:
        confidence.append(0.97)

    elif score >= 60:
        confidence.append(0.92)

    elif score >= 40:
        confidence.append(0.87)

    elif score >= 20:
        confidence.append(0.80)

    else:
        confidence.append(0.75)

df["Confidence"] = confidence

# ----------------------------------------------------
# Recommendation
# ----------------------------------------------------

def recommendation(row):

    if row["Final_Status"] == "Healthy":
        return "Continue Monitoring"

    if row["Final_Status"] == "Observation":
        return "Observe Behaviour"

    if row["Final_Status"] == "At Risk":
        return "Inspect Animal"

    return "Veterinary Check Recommended"

df["Recommendation"] = df.apply(
    recommendation,
    axis=1
)

# ----------------------------------------------------
# Save
# ----------------------------------------------------

OUTPUT = "generated/final_prediction.csv"

df.to_csv(
    OUTPUT,
    index=False
)

print()
print(df.head())

print()
print("=" * 60)
print("Decision Fusion Completed")
print("=" * 60)

print()

print("Saved:")
print(OUTPUT)

print()

print(df["Final_Status"].value_counts())