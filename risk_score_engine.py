"""
===============================================================
risk_score_engine.py

Risk Scoring Engine for Cattle Health Monitoring

This module computes a composite risk score based on multiple
physiological indicators and environmental conditions.
===============================================================
"""

import os
import random
import numpy as np
import pandas as pd

from config import *

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

OUTPUT_FILE = "generated/risk_scores.csv"

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------

def load_data():
    if not os.path.exists("generated/healthy_baseline.csv"):
        raise FileNotFoundError("Generated baseline dataset not found")

    if not os.path.exists("generated/physiology_dataset.csv"):
        raise FileNotFoundError("Generated physiology dataset not found")

    baseline_df = pd.read_csv("generated/healthy_baseline.csv")
    physiology_df = pd.read_csv("generated/physiology_dataset.csv")

    return baseline_df, physiology_df


# ---------------------------------------------------------------
# Compute Risk Score
# ---------------------------------------------------------------

def compute_risk_score(row):
    score = 0.0

    # Heart rate deviation
    if row["Heart_Rate_BPM"] > 85:
        score += 0.30
    elif row["Heart_Rate_BPM"] > 75:
        score += 0.15

    # Body temperature deviation
    if row["Body_Temperature_C"] > 39.5:
        score += 0.25
    elif row["Body_Temperature_C"] > 39.0:
        score += 0.15

    # Rumination deviation
    if row["Rumination_Min"] < 350:
        score += 0.20
    elif row["Rumination_Min"] < 450:
        score += 0.10

    # Activity deviation
    if row["Activity_Index"] < 50:
        score += 0.15
    elif row["Activity_Index"] < 65:
        score += 0.08

    # THI stress
    if row["THI"] > 89:
        score += 0.35
    elif row["THI"] > 79:
        score += 0.20

    # Drinking deviation
    if row["Drinking_Min"] > 25:
        score += 0.10
    elif row["Drinking_Min"] > 20:
        score += 0.05

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------
# Create Risk Scoring Dataset
# ---------------------------------------------------------------

def generate_risk_scores():
    baseline_df, physiology_df = load_data()

    combined_df = pd.concat([baseline_df, physiology_df], ignore_index=True)

    combined_df["Risk_Score"] = combined_df.apply(compute_risk_score, axis=1)

    os.makedirs("generated", exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("Risk scores generated successfully")
    print(combined_df.head())

    return combined_df


if __name__ == "__main__":
    generate_risk_scores()
