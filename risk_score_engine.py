"""
=========================================================
Literature-based Risk Score Engine
=========================================================

Input
------
generated/sensor_data.csv

Output
-------
generated/risk_scores.csv
"""

import os
import pandas as pd

INPUT = "generated/sensor_data.csv"
OUTPUT = "generated/risk_scores.csv"

df = pd.read_csv(INPUT)


def calculate_risk(row):

    score = 0

    # ------------------------------------
    # Heart Rate (20)
    # Normal: 48-84 bpm
    # ------------------------------------
    hr = row["Heart_Rate"]

    if hr < 48 or hr > 84:
        score += 20
    elif hr < 55 or hr > 80:
        score += 10

    # ------------------------------------
    # Body Temperature (20)
    # Normal: 38.5-39.5
    # ------------------------------------
    temp = row["Body_Temperature"]

    if temp < 38.5 or temp > 40.5:
        score += 20
    elif temp > 39.5:
        score += 10

    # ------------------------------------
    # Rumination (15)
    # ------------------------------------
    rum = row["Rumination_Time"]

    if rum < 250:
        score += 15
    elif rum < 350:
        score += 8

    # ------------------------------------
    # Eating (10)
    # ------------------------------------
    eat = row["Eating_Time"]

    if eat < 180:
        score += 10
    elif eat < 240:
        score += 5

    # ------------------------------------
    # Walking (10)
    # ------------------------------------
    walk = row["Walking_Time"]

    if walk < 30:
        score += 10
    elif walk < 45:
        score += 5

    # ------------------------------------
    # Lying (10)
    # ------------------------------------
    lying = row["Lying_Time"]

    if lying > 900:
        score += 10
    elif lying > 750:
        score += 5

    # ------------------------------------
    # THI (10)
    # ------------------------------------
    thi = row["THI"]

    if thi > 89:
        score += 10
    elif thi > 79:
        score += 5

    # ------------------------------------
    # Activity Index (5)
    # ------------------------------------
    activity = row["Activity_Index"]

    if activity < 0.30:
        score += 5
    elif activity < 0.50:
        score += 2

    return score


df["Risk_Score"] = df.apply(calculate_risk, axis=1)


def risk_level(score):

    if score < 30:
        return "Healthy"

    elif score < 60:
        return "At Risk"

    else:
        return "Diseased"


df["Risk_Level"] = df["Risk_Score"].apply(risk_level)

os.makedirs("generated", exist_ok=True)

df.to_csv(OUTPUT, index=False)

print("="*60)
print("Risk Score Engine Completed")
print("="*60)

print(df[["Cow_ID", "Risk_Score", "Risk_Level"]].head())

print()
print("Saved:")
print(OUTPUT)