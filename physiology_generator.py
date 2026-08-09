# ==========================================================
# physiology_dataset.py
#
# Phase-1
# Physiology Dataset Generator (Version-2)
#
# Research Grade Synthetic Dataset
#
# Output:
# generated/physiology_dataset.csv
#
# Author:
# AI-Powered Multi-Modal Cattle Health Monitoring System
# ==========================================================

import os
import random
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Random Seed
# ----------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------

OUTPUT_FILE = "generated/physiology_dataset.csv"

# ----------------------------------------------------------
# Dataset Configuration
# ----------------------------------------------------------

NUM_COWS = 500

DAYS = 120

SAMPLES_PER_DAY = 144

TIME_INTERVAL = 10      # minutes

TOTAL_SAMPLES = NUM_COWS * DAYS * SAMPLES_PER_DAY

HOURS = np.arange(0, 24, TIME_INTERVAL / 60)

print("=" * 60)
print("Generating Physiology Dataset V2")
print("=" * 60)

print(f"Cows          : {NUM_COWS}")
print(f"Days          : {DAYS}")
print(f"Samples / Day : {SAMPLES_PER_DAY}")
print(f"Total Samples : {TOTAL_SAMPLES}")
print()

# ----------------------------------------------------------
# Feature Limits
# ----------------------------------------------------------

LIMITS = {

    "Heart_Rate_BPM": (45, 130),

    "Body_Temperature_C": (37.5, 42.5),

    "Rumination_Min": (100, 700),

    "Eating_Hr": (0, 8),

    "Drinking_Min": (1, 40),

    "Walking_km": (0, 12),

    "Standing_Hr": (0, 24),

    "Lying_Hr": (0, 24),

    "Activity_Index": (0, 100),

    "Ambient_Temperature_C": (10, 42),

    "Relative_Humidity": (20, 100)

}

# ----------------------------------------------------------
# Cow Baseline Generator
# Every Cow Has Different Normal Values
# ----------------------------------------------------------

cow_profiles = {}

for cow in range(1, NUM_COWS + 1):

    cow_id = f"COW_{cow:03d}"

    cow_profiles[cow_id] = {

        "hr_base":

            np.random.normal(
                62,
                4
            ),

        "temp_base":

            np.random.normal(
                38.6,
                0.18
            ),

        "rumination_base":

            np.random.normal(
                470,
                30
            ),

        "eating_base":

            np.random.normal(
                5.6,
                0.45
            ),

        "drinking_base":

            np.random.normal(
                15,
                2
            ),

        "walking_base":

            np.random.normal(
                5.8,
                0.7
            ),

        "standing_base":

            np.random.normal(
                8.2,
                0.8
            ),

        "lying_base":

            np.random.normal(
                8.4,
                0.7
            ),

        "activity_base":

            np.random.normal(
                72,
                7
            )

    }

print("Cow Baselines Created Successfully")
print()

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def clip_value(value, feature):

    low, high = LIMITS[feature]

    return np.clip(value, low, high)


def add_sensor_noise(value, std):

    """
    Simulate Sensor Error
    """

    return value + np.random.normal(
        0,
        std
    )


def chance(probability):

    """
    probability = 0~1
    """

    return random.random() < probability


print("Utility Functions Ready")
print("=" * 60)
print()

# ----------------------------------------------------------
# THI Calculation
# ----------------------------------------------------------

def calculate_thi(temp, humidity):

    thi = (
        (1.8 * temp + 32)
        -
        (
            (0.55 - 0.0055 * humidity)
            *
            (1.8 * temp - 26)
        )
    )

    return round(thi, 2)


# ----------------------------------------------------------
# Circadian Rhythm
# ----------------------------------------------------------

def circadian_factor(hour):

    """
    Simulate natural biological rhythm.
    Highest activity around afternoon.
    Lowest around midnight.
    """

    return np.sin(
        (hour - 6)
        *
        np.pi
        /
        12
    )


# ----------------------------------------------------------
# Daily Weather Generator
# ----------------------------------------------------------

def generate_environment(day):

    season = np.sin(
        2 * np.pi * day / DAYS
    )

    ambient_temp = np.random.normal(
        25 + season * 6,
        2
    )

    humidity = np.random.normal(
        68 - season * 8,
        6
    )

    ambient_temp = clip_value(
        ambient_temp,
        "Ambient_Temperature_C"
    )

    humidity = clip_value(
        humidity,
        "Relative_Humidity"
    )

    thi = calculate_thi(
        ambient_temp,
        humidity
    )

    return (
        ambient_temp,
        humidity,
        thi
    )


# ----------------------------------------------------------
# Daily Random Stress Event
# ----------------------------------------------------------

def generate_daily_stress():

    """
    Healthy cows may occasionally become stressed.

    Transportation

    Vaccination

    Loud sound

    Rain

    Human handling

    etc.
    """

    if chance(0.12):

        return np.random.uniform(
            0.4,
            1.0
        )

    return 0.0

# ----------------------------------------------------------
# Generate One Physiological Sample
# ----------------------------------------------------------

def generate_sample(
        profile,
        hour,
        ambient_temp,
        humidity,
        thi,
        stress_level
):

    rhythm = circadian_factor(hour)

    heat_effect = max(
        0,
        (thi - 72)
        / 25
    )

    # --------------------------------------------
    # Heart Rate
    # --------------------------------------------

    heart_rate = (
        profile["hr_base"]
        +
        rhythm * 2.5
        +
        heat_effect * 5
        +
        stress_level * 8
    )

    # --------------------------------------------
    # Body Temperature
    # --------------------------------------------

    body_temp = (
        profile["temp_base"]
        +
        heat_effect * 0.45
        +
        stress_level * 0.25
    )

    # --------------------------------------------
    # Activity
    # --------------------------------------------

    activity = (
        profile["activity_base"]
        +
        rhythm * 12
        -
        heat_effect * 10
        -
        stress_level * 8
    )

    # --------------------------------------------
    # Walking
    # --------------------------------------------

    walking = (
        profile["walking_base"]
        +
        rhythm * 0.8
        -
        heat_effect * 0.5
    )

    # --------------------------------------------
    # Rumination
    # --------------------------------------------

    rumination = (
        profile["rumination_base"]
        -
        heat_effect * 35
        -
        stress_level * 30
    )

    # --------------------------------------------
    # Eating
    # --------------------------------------------

    eating = (
        profile["eating_base"]
        -
        heat_effect * 0.7
        -
        stress_level * 0.6
    )

    # --------------------------------------------
    # Drinking
    # --------------------------------------------

    drinking = (
        profile["drinking_base"]
        +
        heat_effect * 3
    )

    # --------------------------------------------
    # Standing
    # --------------------------------------------

    standing = (
        profile["standing_base"]
        +
        rhythm * 0.5
    )

    # --------------------------------------------
    # Lying
    # --------------------------------------------

    lying = (
        profile["lying_base"]
        -
        rhythm * 0.5
    )

    heart_rate = add_sensor_noise(
        heart_rate,
        1.4
    )

    body_temp = add_sensor_noise(
        body_temp,
        0.08
    )

    rumination = add_sensor_noise(
        rumination,
        10
    )

    eating = add_sensor_noise(
        eating,
        0.18
    )

    drinking = add_sensor_noise(
        drinking,
        0.8
    )

    walking = add_sensor_noise(
        walking,
        0.18
    )

    standing = add_sensor_noise(
        standing,
        0.20
    )

    lying = add_sensor_noise(
        lying,
        0.20
    )

    activity = add_sensor_noise(
        activity,
        2
    )
    # ----------------------------------------------------------
# Healthy Outlier Injection
# ----------------------------------------------------------

    if chance(0.03):

        feature = random.choice([

            "Heart_Rate_BPM",
            "Body_Temperature_C",
            "Rumination_Min",
            "Eating_Hr",
            "Drinking_Min",
            "Walking_km",
            "Standing_Hr",
            "Lying_Hr",
            "Activity_Index"

        ])

        if feature == "Heart_Rate_BPM":
            heart_rate += random.uniform(-10,10)

        elif feature == "Body_Temperature_C":
            body_temp += random.uniform(-0.5,0.6)

        elif feature == "Rumination_Min":
            rumination += random.uniform(-80,80)

        elif feature == "Eating_Hr":
            eating += random.uniform(-1.2,1.2)

        elif feature == "Drinking_Min":
            drinking += random.uniform(-5,5)

        elif feature == "Walking_km":
            walking += random.uniform(-2,2)

        elif feature == "Standing_Hr":
            standing += random.uniform(-2,2)

        elif feature == "Lying_Hr":
            lying += random.uniform(-2,2)

        elif feature == "Activity_Index":
            activity += random.uniform(-20,20)

    # ----------------------------------------------------------
    # Missing Sensor Reading
    # ----------------------------------------------------------

    if chance(0.01):

        sensor = random.choice([

            "heart",
            "temp",
            "rumination",
            "activity"

        ])

        if sensor=="heart":
            heart_rate=np.nan

        elif sensor=="temp":
            body_temp=np.nan

        elif sensor=="rumination":
            rumination=np.nan

        elif sensor=="activity":
            activity=np.nan

    # ----------------------------------------------------------
    # Sensor Drift
    # ----------------------------------------------------------

    drift = np.random.normal(0,0.3)

    heart_rate += drift

    body_temp += drift*0.02

    activity += drift*1.2

    # ----------------------------------------------------------
    # Behaviour Randomness
    # ----------------------------------------------------------

    walking *= np.random.uniform(0.92,1.08)

    standing *= np.random.uniform(0.94,1.06)

    lying *= np.random.uniform(0.94,1.06)

    eating *= np.random.uniform(0.90,1.08)

    rumination *= np.random.uniform(0.90,1.08)

    activity *= np.random.uniform(0.90,1.08)
    # ----------------------------------------------------------
    # Weather Spike
    # ----------------------------------------------------------

    if chance(0.02):

        ambient_temp += np.random.uniform(4,8)

        humidity += np.random.uniform(-10,10)

        thi = calculate_thi(
            ambient_temp,
            humidity
        )
    # ----------------------------------------------------------
    # Temporary Heat Stress
    # ----------------------------------------------------------

    if thi > 80 and chance(0.35):

        heart_rate += np.random.uniform(3,8)

        drinking += np.random.uniform(2,5)

        eating -= np.random.uniform(0.4,1)

        activity -= np.random.uniform(5,12)

    # ----------------------------------------------------------
    # Exercise Event
    # ----------------------------------------------------------

    if chance(0.04):

        walking += np.random.uniform(1,3)

        heart_rate += np.random.uniform(4,10)

        activity += np.random.uniform(6,15)

    heart_rate=clip_value(heart_rate,"Heart_Rate_BPM")

    body_temp=clip_value(body_temp,"Body_Temperature_C")

    rumination=clip_value(rumination,"Rumination_Min")

    eating=clip_value(eating,"Eating_Hr")

    drinking=clip_value(drinking,"Drinking_Min")

    walking=clip_value(walking,"Walking_km")

    standing=clip_value(standing,"Standing_Hr")

    lying=clip_value(lying,"Lying_Hr")

    activity=clip_value(activity,"Activity_Index")
    return {

        "Heart_Rate_BPM":
            clip_value(
                round(heart_rate,2),
                "Heart_Rate_BPM"
            ),

        "Body_Temperature_C":
            clip_value(
                round(body_temp,2),
                "Body_Temperature_C"
            ),

        "Rumination_Min":
            clip_value(
                round(rumination,2),
                "Rumination_Min"
            ),

        "Eating_Hr":
            clip_value(
                round(eating,2),
                "Eating_Hr"
            ),

        "Drinking_Min":
            clip_value(
                round(drinking,2),
                "Drinking_Min"
            ),

        "Walking_km":
            clip_value(
                round(walking,2),
                "Walking_km"
            ),

        "Standing_Hr":
            clip_value(
                round(standing,2),
                "Standing_Hr"
            ),

        "Lying_Hr":
            clip_value(
                round(lying,2),
                "Lying_Hr"
            ),

        "Activity_Index":
            clip_value(
                round(activity,2),
                "Activity_Index"
            ),

        "Ambient_Temperature_C":
            round(ambient_temp,2),

        "Relative_Humidity":
            round(humidity,2),

        "THI":
            round(thi,2)

    }

# ==========================================================
# Generate Complete Dataset
# ==========================================================

print("=" * 60)
print("Generating Physiology Dataset")
print("=" * 60)

records = []

sample_id = 1


def generate_cow_profile(cow_id):
    return {
        "hr_base": np.random.normal(62, 4),
        "temp_base": np.random.normal(38.6, 0.18),
        "rumination_base": np.random.normal(470, 30),
        "eating_base": np.random.normal(5.6, 0.45),
        "drinking_base": np.random.normal(15, 2),
        "walking_base": np.random.normal(5.8, 0.7),
        "standing_base": np.random.normal(8.2, 0.8),
        "lying_base": np.random.normal(8.4, 0.7),
        "activity_base": np.random.normal(72, 7),
    }


for cow in range(1, NUM_COWS + 1):

    cow_id = f"COW_{cow:03d}"

    profile = generate_cow_profile(cow_id)

    print(f"Generating {cow_id}")

    for day in range(1, DAYS + 1):

        (
            ambient_temp,
            humidity,
            thi
        ) = generate_environment(day)

        stress = generate_daily_stress()

        for hour in HOURS:

            sample = generate_sample(

                profile,
                hour,
                ambient_temp,
                humidity,
                thi,
                stress

            )

            records.append({

                "Sample_ID": sample_id,

                "Cow_ID": cow_id,

                "Day": day,

                "Hour": round(hour,2),

                **sample

            })

            sample_id += 1

print()
print("Dataset Generation Completed")
# ==========================================================
# Convert to DataFrame
# ==========================================================

df = pd.DataFrame(records)

print()
print(df.head())

print()
print(df.shape)
# ==========================================================
# Shuffle Dataset
# ==========================================================

df = df.sample(

    frac=1,

    random_state=42

).reset_index(drop=True)
# ==========================================================
# Save Dataset
# ==========================================================

os.makedirs(

    "generated",

    exist_ok=True

)

OUTPUT_FILE = "generated/physiology_dataset.csv"

df.to_csv(

    OUTPUT_FILE,

    index=False

)

print()
print("=" * 60)
print("Physiology Dataset Saved Successfully")
print("=" * 60)

print()

print("Output File :")

print(OUTPUT_FILE)

print()

print("Total Samples :", len(df))

print()

print("Columns")

print(df.columns.tolist())
