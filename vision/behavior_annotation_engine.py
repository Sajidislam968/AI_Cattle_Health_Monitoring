# ==========================================================
# behavior_annotation_engine.py
#
# Phase-8
# Vision AI
#
# Automatic Behavior Annotation Engine
#
# Input
# -----
# vision/output/tracking_results.csv
#
# Output
# ------
# vision/output/behavior_dataset/behavior_dataset.csv
#
# ==========================================================

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------

TRACKING_FILE = "vision/output/tracking_results.csv"

OUTPUT_FILE = "vision/output/behavior_dataset/behavior_dataset.csv"

FRAME_RATE = 30

WINDOW_SIZE = 15

# ----------------------------------------------------------

print("=" * 60)
print("Behavior Annotation Engine")
print("=" * 60)

if not os.path.exists(TRACKING_FILE):
    raise FileNotFoundError(TRACKING_FILE)

df = pd.read_csv(TRACKING_FILE)

print()
print("Tracking Samples :", len(df))

df = df.sort_values(
    ["Cow_ID", "Frame"]
).reset_index(drop=True)

records = []
# ==========================================================
# Calculate Motion Features
# ==========================================================

for cow_id in df["Cow_ID"].unique():

    cow = df[df["Cow_ID"] == cow_id].copy()

    cow["DX"] = cow["Center_X"].diff().fillna(0)
    cow["DY"] = cow["Center_Y"].diff().fillna(0)

    cow["Movement"] = np.sqrt(
        cow["DX"] ** 2 +
        cow["DY"] ** 2
    )

    cow["Speed"] = (
        cow["Movement"] * FRAME_RATE
    )

    cow["BBoxArea"] = (
        cow["Width"] *
        cow["Height"]
    )

    cow["AspectRatio"] = (
        cow["Height"] /
        (cow["Width"] + 1e-6)
    )

    cow.loc[:, "SmoothSpeed"] = (

        cow["Speed"]

        .rolling(
            WINDOW_SIZE,
            min_periods=1
        )

        .mean()

    )

    cow.loc[:, "SmoothArea"] = (

        cow["BBoxArea"]

        .rolling(
            WINDOW_SIZE,
            min_periods=1
        )

        .mean()

    )

    labels = []

    for _, row in cow.iterrows():

        speed = row["SmoothSpeed"]
        area = row["SmoothArea"]
        ratio = row["AspectRatio"]

        # ----------------------------------
        # Rule-based Annotation
        # ----------------------------------

        if ratio > 1.45 and speed < 5:

            label = "Standing"

        elif ratio < 0.85:

            label = "Lying"

        elif speed > 35:

            label = "Walking"

        elif speed < 12 and area > 12000:

            label = "Eating"

        else:

            label = "Rumination"

        labels.append(label)

    cow["Behavior_Label"] = labels

    records.append(cow)

# ==========================================================
# Merge
# ==========================================================

behavior_df = pd.concat(
    records,
    ignore_index=True
)

behavior_df.to_csv(

    OUTPUT_FILE,

    index=False

)

print()
print("=" * 60)
print("Behavior Annotation Completed")
print("=" * 60)

print()

print("Output File")
print(OUTPUT_FILE)

print()

print("Behavior Distribution")

print(

    behavior_df[
        "Behavior_Label"
    ].value_counts()

)

print()

print("Total Samples")

print(len(behavior_df))