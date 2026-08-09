"""
=========================================================
behavior_generator.py

Generate Behavioral Dataset

Input:
    generated/environment_dataset.csv

Output:
    generated/behavior_dataset.csv

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
# Disease Assignment
# -------------------------------------------------------

DISEASE_STAGES = [
    "Healthy",
    "At Risk",
    "Critical"
]

DISEASE_PROB = [
    0.70,
    0.20,
    0.10
]

# -------------------------------------------------------
# Helper
# -------------------------------------------------------

def clip(value, low, high):
    return max(low, min(high, value))


def assign_stage():

    return random.choices(
        DISEASE_STAGES,
        weights=DISEASE_PROB,
        k=1
    )[0]

# -------------------------------------------------------
# Disease Rules
# -------------------------------------------------------

def apply_behavior_rules(row, stage):

    rumination = row["Rumination_Min"]
    eating = row["Eating_Hr"]
    drinking = row["Drinking_Min"]
    walking = row["Walking_km"]
    standing = row["Standing_Hr"]
    lying = row["Lying_Hr"]
    activity = row["Activity_Index"]

    thi = row["THI"]

    # ----------------------------------
    # Heat Stress Rules
    # ----------------------------------

    if thi >= THI_SAFE:

        drinking += random.uniform(0.5, 2.0)

        standing += random.uniform(0.2, 0.8)

        lying -= random.uniform(0.2, 0.8)

        activity -= random.uniform(1, 3)

    # ----------------------------------
    # Disease Rules
    # ----------------------------------

    if stage == "Healthy":

        rumination += np.random.normal(0, 5)

        eating += np.random.normal(0, 0.10)

        drinking += np.random.normal(0, 0.30)

        walking += np.random.normal(0, 0.20)

        standing += np.random.normal(0, 0.15)

        lying += np.random.normal(0, 0.15)

        activity += np.random.normal(0, 1)

    elif stage == "At Risk":

        rumination *= random.uniform(0.88, 0.95)

        eating *= random.uniform(0.85, 0.95)

        walking *= random.uniform(0.82, 0.92)

        activity *= random.uniform(0.85, 0.92)

        standing *= random.uniform(1.03, 1.08)

        lying *= random.uniform(1.02, 1.08)

        drinking *= random.uniform(0.92, 1.05)

    else:

        rumination *= random.uniform(0.55, 0.75)

        eating *= random.uniform(0.50, 0.70)

        walking *= random.uniform(0.45, 0.70)

        activity *= random.uniform(0.50, 0.70)

        standing *= random.uniform(1.10, 1.25)

        lying *= random.uniform(1.10, 1.30)

        drinking *= random.uniform(0.80, 1.10)

    # ----------------------------------
    # Clipping
    # ----------------------------------

    rumination = clip(rumination, 150, 650)

    eating = clip(eating, 1.0, 6.5)

    drinking = clip(drinking, 3, 30)

    walking = clip(walking, 0.5, 10)

    standing = clip(standing, 4, 18)

    lying = clip(lying, 4, 18)

    activity = clip(activity, 20, 100)

    return (
        round(rumination, 2),
        round(eating, 2),
        round(drinking, 2),
        round(walking, 2),
        round(standing, 2),
        round(lying, 2),
        round(activity, 2)
    )

# -------------------------------------------------------
# Risk Score
# -------------------------------------------------------

def calculate_behavior_score(stage):

    if stage == "Healthy":

        return random.randint(0, 30)

    elif stage == "At Risk":

        return random.randint(31, 70)

    else:

        return random.randint(71, 100)

# -------------------------------------------------------
# Generator
# -------------------------------------------------------

def generate_behavior():

    df = pd.read_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "environment_dataset.csv"
        )
    )

    disease_stage = []

    behavior_score = []

    rumination_list = []

    eating_list = []

    drinking_list = []

    walking_list = []

    standing_list = []

    lying_list = []

    activity_list = []

    previous_stage = {}

    for index, row in df.iterrows():

        cow = row["Cow_ID"]

        day = row["Day"]

        if cow not in previous_stage:

            stage = assign_stage()

            previous_stage[cow] = stage

        else:

            stage = previous_stage[cow]

            if day % 10 == 0 and random.random() < 0.10:

                stage = assign_stage()

                previous_stage[cow] = stage

        (
            rum,
            eat,
            drink,
            walk,
            stand,
            lie,
            act
        ) = apply_behavior_rules(
            row,
            stage
        )

        rumination_list.append(rum)

        eating_list.append(eat)

        drinking_list.append(drink)

        walking_list.append(walk)

        standing_list.append(stand)

        lying_list.append(lie)

        activity_list.append(act)

        disease_stage.append(stage)

        behavior_score.append(
            calculate_behavior_score(stage)
        )

    df["Rumination_Min"] = rumination_list

    df["Eating_Hr"] = eating_list

    df["Drinking_Min"] = drinking_list

    df["Walking_km"] = walking_list

    df["Standing_Hr"] = standing_list

    df["Lying_Hr"] = lying_list

    df["Activity_Index"] = activity_list

    df["Behavior_State"] = disease_stage

    df["Behavior_Risk_Score"] = behavior_score

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "behavior_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("-------------------------------------")
    print("Behavior Dataset Generated")
    print("-------------------------------------")
    print(df.head())
    print()
    print(output_file)

    return df

# -------------------------------------------------------
# Run
# -------------------------------------------------------

if __name__ == "__main__":

    generate_behavior()