"""
=========================================================
cow_profile_generator.py

Generate Individual Cow Profiles

Output:
generated/cow_profiles.csv
=========================================================
"""

import os
import random
import pandas as pd

from config import *
from config import PROFILE_FILE

# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

random.seed(RANDOM_SEED)

# -------------------------------------------------------
# Categories
# -------------------------------------------------------

BREEDS = [
    "Holstein",
    "Jersey",
    "Crossbreed"
]

LACTATION_STAGE = [
    "Early",
    "Mid",
    "Late"
]

# -------------------------------------------------------
# Healthy Baseline Ranges
# (Based on selected literature)
# -------------------------------------------------------

HEART_RATE_RANGE = (52, 80)          # BPM

BODY_TEMP_RANGE = (38.5, 39.4)       # Celsius

RUMINATION_RANGE = (420, 600)        # min/day

EATING_RANGE = (4.0, 6.0)            # hr/day

DRINKING_RANGE = (8, 20)             # min/day

WALKING_RANGE = (3.5, 8.0)           # km/day

STANDING_RANGE = (8.0, 12.0)         # hr/day

LYING_RANGE = (8.0, 14.0)            # hr/day

ACTIVITY_RANGE = (65, 100)           # Index

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def rand_float(low, high, digit=2):
    return round(random.uniform(low, high), digit)


def generate_profile(cow_id):

    breed = random.choice(BREEDS)

    age = random.randint(2, 9)

    parity = random.randint(1, 5)

    lactation = random.choice(LACTATION_STAGE)

    if breed == "Holstein":
        weight = random.randint(520, 700)

    elif breed == "Jersey":
        weight = random.randint(350, 500)

    else:
        weight = random.randint(420, 620)

    profile = {

        "Cow_ID": f"COW_{cow_id:03d}",

        "Breed": breed,

        "Age_Years": age,

        "Weight_kg": weight,

        "Parity": parity,

        "Lactation_Stage": lactation,

        "Baseline_HeartRate_BPM":
            rand_float(*HEART_RATE_RANGE),

        "Baseline_BodyTemp_C":
            rand_float(*BODY_TEMP_RANGE),

        "Baseline_Rumination_Min":
            rand_float(*RUMINATION_RANGE),

        "Baseline_Eating_Hr":
            rand_float(*EATING_RANGE),

        "Baseline_Drinking_Min":
            rand_float(*DRINKING_RANGE),

        "Baseline_Walking_km":
            rand_float(*WALKING_RANGE),

        "Baseline_Standing_Hr":
            rand_float(*STANDING_RANGE),

        "Baseline_Lying_Hr":
            rand_float(*LYING_RANGE),

        "Baseline_Activity":
            rand_float(*ACTIVITY_RANGE)

    }

    return profile


# -------------------------------------------------------
# Main Generator
# -------------------------------------------------------

def generate_cow_profiles():

    profiles = []

    for i in range(1, NUM_CATTLE + 1):

        profiles.append(generate_profile(i))

    df = pd.DataFrame(profiles)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df.to_csv(PROFILE_FILE, index=False)

    print("-------------------------------------")
    print("Cow Profile Generation Completed")
    print("-------------------------------------")
    print(df.head())

    print()
    print("Saved to:")
    print(PROFILE_FILE)

    return df


# -------------------------------------------------------
# Run
# -------------------------------------------------------

if __name__ == "__main__":

    generate_cow_profiles()