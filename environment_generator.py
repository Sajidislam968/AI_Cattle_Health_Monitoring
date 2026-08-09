"""
=========================================================
environment_generator.py

Generate Environmental Conditions

Input:
    generated/healthy_baseline.csv

Output:
    generated/environment_dataset.csv

=========================================================
"""

import os
import random
import numpy as np
import pandas as pd

from config import *

# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------
# THI Calculation
# -------------------------------------------------------

def calculate_thi(temp, humidity):

    thi = (
        (1.8 * temp + 32)
        - (
            (0.55 - 0.0055 * humidity)
            * (1.8 * temp - 26)
        )
    )

    return round(thi, 2)

# -------------------------------------------------------
# THI Level
# -------------------------------------------------------

def get_thi_level(thi):

    if thi < THI_SAFE:
        return "Normal"

    elif thi < THI_MODERATE:
        return "Mild"

    elif thi < THI_SEVERE:
        return "Moderate"

    else:
        return "Severe"

# -------------------------------------------------------
# Day / Night Temperature Pattern
# -------------------------------------------------------

def generate_temperature(hour):

    if 5 <= hour < 10:

        temp = random.uniform(20, 25)

    elif 10 <= hour < 16:

        temp = random.uniform(28, 38)

    elif 16 <= hour < 20:

        temp = random.uniform(24, 30)

    else:

        temp = random.uniform(18, 23)

    temp += np.random.normal(0, 0.5)

    return round(temp, 2)

# -------------------------------------------------------
# Humidity Pattern
# -------------------------------------------------------

def generate_humidity(hour):

    if 10 <= hour < 16:

        humidity = random.uniform(40, 65)

    else:

        humidity = random.uniform(55, 90)

    humidity += np.random.normal(0, 2)

    humidity = max(30, min(95, humidity))

    return round(humidity, 2)

# -------------------------------------------------------
# Generate Environment Dataset
# -------------------------------------------------------

def generate_environment():

    df = pd.read_csv(BASELINE_FILE)

    ambient_temp_list = []

    humidity_list = []

    thi_list = []

    heat_level = []

    heat_flag = []

    for _, row in df.iterrows():

        hour = row["Hour"]

        temp = generate_temperature(hour)

        humidity = generate_humidity(hour)

        thi = calculate_thi(
            temp,
            humidity
        )

        level = get_thi_level(thi)

        ambient_temp_list.append(temp)

        humidity_list.append(humidity)

        thi_list.append(thi)

        heat_level.append(level)

        heat_flag.append(
            1 if thi >= THI_SAFE else 0
        )

    df["Ambient_Temperature_C"] = ambient_temp_list

    df["Relative_Humidity"] = humidity_list

    df["THI"] = thi_list

    df["Heat_Stress_Level"] = heat_level

    df["Heat_Stress_Flag"] = heat_flag

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "environment_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("------------------------------------")
    print("Environment Dataset Generated")
    print("------------------------------------")
    print(df.head())
    print()
    print(output_file)

    return df

# -------------------------------------------------------
# Run
# -------------------------------------------------------

if __name__ == "__main__":

    generate_environment()