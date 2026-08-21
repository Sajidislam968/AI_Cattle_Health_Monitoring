"""
============================================================
Synthetic Sensor Generator
============================================================

Input
------
vision/output/behavior_summary.csv

Output
------
generated/sensor_data.csv

Creates realistic IoT sensor values based on
predicted behaviour.

Used for testing the Decision Fusion pipeline.
"""

import random
import os
import pandas as pd

INPUT_FILE = "vision/output/behavior_summary.csv"
OUTPUT_FILE = "generated/sensor_data.csv"

print("="*60)
print("Generating Synthetic Sensor Data")
print("="*60)

df = pd.read_csv(INPUT_FILE)

sensor_rows = []

for _, row in df.iterrows():

    cow_id = row["Cow_ID"]
    behavior = str(row["Dominant_Behavior"]).lower()

    # ---------------------------------------------------
    # Normal Defaults
    # ---------------------------------------------------

    heart_rate = random.uniform(55, 80)

    body_temp = random.uniform(38.4, 39.2)

    humidity = random.uniform(55, 80)

    ambient_temp = random.uniform(24, 35)

    thi = random.uniform(70, 85)

    activity = random.uniform(0.40, 0.75)

    eating_time = random.uniform(240, 360)         # min/day

    rumination_time = random.uniform(350, 550)     # min/day

    walking_time = random.uniform(40, 120)         # min/day

    lying_time = random.uniform(500, 750)          # min/day

    # ---------------------------------------------------
    # Behaviour Adjustment
    # ---------------------------------------------------

    if behavior == "standing":

        activity = random.uniform(0.50, 0.70)
        walking_time = random.uniform(40, 80)

    elif behavior == "walking":

        activity = random.uniform(0.80, 1.00)
        walking_time = random.uniform(120, 240)
        heart_rate += random.uniform(5, 10)

    elif behavior == "eating":

        eating_time = random.uniform(320, 420)
        activity = random.uniform(0.30, 0.55)

    elif behavior == "lying":

        lying_time = random.uniform(700, 900)
        activity = random.uniform(0.10, 0.30)

    elif behavior == "rumination":

        rumination_time = random.uniform(500, 700)
        lying_time = random.uniform(650, 850)
        activity = random.uniform(0.10, 0.25)

    sensor_rows.append({

        "Cow_ID": cow_id,

        "Heart_Rate": round(heart_rate,2),

        "Body_Temperature": round(body_temp,2),

        "Eating_Time": round(eating_time,1),

        "Rumination_Time": round(rumination_time,1),

        "Walking_Time": round(walking_time,1),

        "Lying_Time": round(lying_time,1),

        "Humidity": round(humidity,1),

        "Ambient_Temperature": round(ambient_temp,1),

        "THI": round(thi,1),

        "Activity_Index": round(activity,2)

    })

sensor_df = pd.DataFrame(sensor_rows)

os.makedirs("generated", exist_ok=True)

sensor_df.to_csv(

    OUTPUT_FILE,

    index=False

)

print()
print(sensor_df.head())

print()

print("Saved:")

print(OUTPUT_FILE)