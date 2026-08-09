"""
=========================================================
baseline_generator.py

Generate Healthy Baseline Dataset

Input:
generated/cow_profiles.csv

Output:
generated/healthy_baseline.csv

=========================================================
"""

import os
import random
from config import BASELINE_FILE
import numpy as np
import pandas as pd

from config import *

# ----------------------------------------------------
# Random Seed
# ----------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def clip(value, low, high):
    return max(low, min(high, value))


def calculate_thi(temp, humidity):
    """
    Temperature Humidity Index (THI)

    Literature-based formula
    """

    thi = (1.8 * temp + 32) - (
        (0.55 - 0.0055 * humidity)
        * (1.8 * temp - 26)
    )

    return round(thi, 2)


def generate_environment():

    ambient_temp = round(
        random.uniform(
            AMBIENT_TEMP_RANGE[0],
            AMBIENT_TEMP_RANGE[1]
        ),
        2
    )

    humidity = round(
        random.uniform(
            HUMIDITY_RANGE[0],
            HUMIDITY_RANGE[1]
        ),
        2
    )

    thi = calculate_thi(
        ambient_temp,
        humidity
    )

    return ambient_temp, humidity, thi


# ----------------------------------------------------
# Baseline Generator
# ----------------------------------------------------

def generate_baseline():

    profiles = pd.read_csv(PROFILE_FILE)

    all_rows = []

    sample = 0

    for _, cow in profiles.iterrows():

        for day in range(1, SIMULATION_DAYS + 1):

            for step in range(SAMPLES_PER_DAY):

                sample += 1

                hour = (
                    step * SAMPLING_INTERVAL_MIN
                ) / 60

                # -----------------------------------------
                # Healthy Circadian Variations
                # -----------------------------------------

                heart_rate = (
                    cow["Baseline_HeartRate_BPM"]
                    + np.random.normal(0, 2)
                )

                body_temp = (
                    cow["Baseline_BodyTemp_C"]
                    + 0.2 * np.sin(
                        2 * np.pi * hour / 24
                    )
                    + np.random.normal(0, 0.05)
                )

                rumination = (
                    cow["Baseline_Rumination_Min"]
                    + np.random.normal(0, 12)
                )

                eating = (
                    cow["Baseline_Eating_Hr"]
                    + np.random.normal(0, 0.20)
                )

                drinking = (
                    cow["Baseline_Drinking_Min"]
                    + np.random.normal(0, 1)
                )

                walking = (
                    cow["Baseline_Walking_km"]
                    + np.random.normal(0, 0.30)
                )

                standing = (
                    cow["Baseline_Standing_Hr"]
                    + np.random.normal(0, 0.25)
                )

                lying = (
                    cow["Baseline_Lying_Hr"]
                    + np.random.normal(0, 0.25)
                )

                activity = (
                    cow["Baseline_Activity"]
                    + np.random.normal(0, 3)
                )

                # -----------------------------------------
                # Clip Healthy Values
                # -----------------------------------------

                heart_rate = clip(
                    heart_rate,
                    48,
                    84
                )

                body_temp = clip(
                    body_temp,
                    38.5,
                    39.5
                )

                rumination = clip(
                    rumination,
                    350,
                    650
                )

                eating = clip(
                    eating,
                    3.5,
                    6.5
                )

                drinking = clip(
                    drinking,
                    5,
                    25
                )

                walking = clip(
                    walking,
                    2,
                    10
                )

                standing = clip(
                    standing,
                    6,
                    14
                )

                lying = clip(
                    lying,
                    6,
                    16
                )

                activity = clip(
                    activity,
                    50,
                    100
                )

                # -----------------------------------------
                # Environment
                # -----------------------------------------

                ambient_temp, humidity, thi = generate_environment()

                # -----------------------------------------
                # Save
                # -----------------------------------------

                all_rows.append({

                    "Sample_ID": sample,

                    "Cow_ID": cow["Cow_ID"],

                    "Day": day,

                    "Hour": round(hour, 2),

                    "Heart_Rate_BPM":
                        round(heart_rate, 2),

                    "Body_Temperature_C":
                        round(body_temp, 2),

                    "Rumination_Min":
                        round(rumination, 2),

                    "Eating_Hr":
                        round(eating, 2),

                    "Drinking_Min":
                        round(drinking, 2),

                    "Walking_km":
                        round(walking, 2),

                    "Standing_Hr":
                        round(standing, 2),

                    "Lying_Hr":
                        round(lying, 2),

                    "Activity_Index":
                        round(activity, 2),

                    "Ambient_Temperature_C":
                        ambient_temp,

                    "Humidity":
                        humidity,

                    "THI":
                        thi

                })

    df = pd.DataFrame(all_rows)

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    df.to_csv(
        BASELINE_FILE,
        index=False
    )

    print("---------------------------------------")
    print("Healthy Baseline Generated")
    print("---------------------------------------")
    print(df.head())
    print()
    print(BASELINE_FILE)

    return df


# ----------------------------------------------------
# Run
# ----------------------------------------------------

if __name__ == "__main__":

    generate_baseline()
