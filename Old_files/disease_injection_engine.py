# ==========================================================
# disease_injection_engine.py
#
# Phase-2
# Realistic Disease Injection Engine
#
# Features
# ----------
# ✓ Disease progression
# ✓ Disease specific symptoms
# ✓ Symptom overlap
# ✓ Clinical anomaly injection
# ✓ Sensor noise
# ✓ Individual cow variation
# ✓ Disease label generation
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
# File Path
# ----------------------------------------------------------

INPUT_FILE = "generated/physiology_dataset.csv"
OUTPUT_FILE = "generated/disease_dataset.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

print("Loading Physiology Dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset Shape :", df.shape)

# ----------------------------------------------------------
# Initialize Labels
# ----------------------------------------------------------

df["Health_Status"] = "Healthy"
df["Disease_Name"] = "None"

# ----------------------------------------------------------
# Disease Classes
# ----------------------------------------------------------

DISEASES = [

    "Mastitis",
    "BRD",
    "Lameness",
    "Heat_Stress"

]

# ----------------------------------------------------------
# Cow Selection
# ----------------------------------------------------------

cow_ids = df["Cow_ID"].unique()

total_cows = len(cow_ids)

# Around 35% become affected

num_sick = int(total_cows * 0.35)

infected_cows = np.random.choice(

    cow_ids,

    size=num_sick,

    replace=False

)

print()

print("Total Cows      :", total_cows)
print("Affected Cows   :", num_sick)

# ----------------------------------------------------------
# Disease Severity
# ----------------------------------------------------------

severity_levels = {

    "Mild": 0.8,

    "Moderate": 1.0,

    "Severe": 1.3

}

severity_prob = [

    0.40,

    0.40,

    0.20

]

# ----------------------------------------------------------
# Clinical Anomaly Probability
# ----------------------------------------------------------
# These are NOT random impossible outliers.
# They simulate stress events, temporary sensor errors,
# handling stress, dehydration, etc.

ANOMALY_RATE = 0.05        # 5%

SENSOR_NOISE = 0.03        # 3%

RECOVERY_RATE = 0.12       # 12%

# ----------------------------------------------------------
# Disease Injection
# ----------------------------------------------------------

for cow in infected_cows:

    cow_data = df[df["Cow_ID"] == cow]

    indices = cow_data.index.tolist()

    total_records = len(indices)

    disease = random.choice(DISEASES)

    severity_name = np.random.choice(

        list(severity_levels.keys()),

        p=severity_prob

    )

    severity = severity_levels[severity_name]

    # Disease starts between 40% - 70% timeline

    disease_start = random.randint(

        int(total_records * 0.40),

        int(total_records * 0.70)

    )

    # Some cows partially recover

    recovery = random.random() < RECOVERY_RATE

    recovery_start = random.randint(

        int(total_records * 0.80),

        int(total_records * 0.95)

    )

    # ------------------------------------------------------
    # Progress through timeline
    # ------------------------------------------------------

    for i, idx in enumerate(indices):

        # Healthy phase

        if i < disease_start:
            continue

        progress = (i - disease_start) / max(
            1,
            total_records - disease_start
        )

        # Recovery phase

        if recovery and i >= recovery_start:

            df.loc[idx, "Health_Status"] = random.choice(
                [
                    "Healthy",
                    "At_Risk"
                ]
            )

            df.loc[idx, "Disease_Name"] = "Recovery"

            continue

        # Disease stage

        if progress < 0.45:

            stage = "At_Risk"

        else:

            stage = "Diseased"

        df.loc[idx, "Health_Status"] = stage

        df.loc[idx, "Disease_Name"] = disease

        # --------------------------------------------------
        # Feature Selection (Symptom Overlap)
        # --------------------------------------------------

        symptoms = [

            "Heart_Rate",

            "Temperature",

            "Rumination",

            "Eating",

            "Drinking",

            "Walking",

            "Standing",

            "Lying",

            "Activity"

        ]

        # Every cow will NOT show every symptom

        active_symptoms = random.sample(

            symptoms,

            random.randint(5, 8)

        )
                # ==================================================
        # Disease Specific Feature Modification
        # ==================================================

        # -----------------------------
        # Heart Rate
        # -----------------------------

        if "Heart_Rate" in active_symptoms:

            if stage == "At_Risk":
                hr_change = random.uniform(2, 8) * severity
            else:
                hr_change = random.uniform(8, 22) * severity

            # BRD and Heat Stress affect HR more
            if disease in ["BRD", "Heat_Stress"]:
                hr_change *= random.uniform(1.10, 1.30)

            df.loc[idx, "Heart_Rate_BPM"] += hr_change


        # -----------------------------
        # Body Temperature
        # -----------------------------

        if "Temperature" in active_symptoms:

            if stage == "At_Risk":
                temp_change = random.uniform(0.10,0.60)
            else:
                temp_change = random.uniform(0.50,1.80)

            if disease in ["Mastitis","BRD"]:
                temp_change *= random.uniform(1.10,1.30)

            df.loc[idx,"Body_Temperature_C"] += temp_change*severity


        # -----------------------------
        # Rumination
        # -----------------------------

        if "Rumination" in active_symptoms:

            if stage=="At_Risk":
                factor=random.uniform(0.82,0.98)
            else:
                factor=random.uniform(0.45,0.82)

            df.loc[idx,"Rumination_Min"]*=factor


        # -----------------------------
        # Eating
        # -----------------------------

        if "Eating" in active_symptoms:

            if stage=="At_Risk":
                factor=random.uniform(0.82,0.98)
            else:
                factor=random.uniform(0.45,0.82)

            df.loc[idx,"Eating_Hr"]*=factor


        # -----------------------------
        # Drinking
        # -----------------------------

        if "Drinking" in active_symptoms:

            if disease=="Heat_Stress":

                factor=random.uniform(1.05,1.35)

            else:

                factor=random.uniform(0.70,0.95)

            df.loc[idx,"Drinking_Min"]*=factor


        # -----------------------------
        # Walking
        # -----------------------------

        if "Walking" in active_symptoms:

            if disease=="Lameness":

                factor=random.uniform(0.20,0.55)

            elif stage=="Diseased":

                factor=random.uniform(0.40,0.75)

            else:

                factor=random.uniform(0.75,0.95)

            df.loc[idx,"Walking_km"]*=factor


        # -----------------------------
        # Standing
        # -----------------------------

        if "Standing" in active_symptoms:

            factor=random.uniform(1.00,1.35)

            df.loc[idx,"Standing_Hr"]*=factor


        # -----------------------------
        # Lying
        # -----------------------------

        if "Lying" in active_symptoms:

            factor=random.uniform(1.00,1.45)

            df.loc[idx,"Lying_Hr"]*=factor


        # -----------------------------
        # Activity
        # -----------------------------

        if "Activity" in active_symptoms:

            if stage=="At_Risk":

                factor=random.uniform(0.75,0.98)

            else:

                factor=random.uniform(0.35,0.80)

            df.loc[idx,"Activity_Index"]*=factor


        # ==================================================
        # Sensor Noise
        # ==================================================

        if random.random() < SENSOR_NOISE:

            df.loc[idx,"Heart_Rate_BPM"] += random.uniform(-4,4)

            df.loc[idx,"Body_Temperature_C"] += random.uniform(-0.20,0.20)

            df.loc[idx,"Rumination_Min"] += random.uniform(-20,20)

            df.loc[idx,"Activity_Index"] += random.uniform(-8,8)


        # ==================================================
        # Clinical Anomaly Injection
        # ==================================================

        if random.random() < ANOMALY_RATE:

            anomaly=random.choice([

                "Stress",

                "Sensor_Error",

                "Heat_Event",

                "Fatigue"

            ])

            if anomaly=="Stress":

                df.loc[idx,"Heart_Rate_BPM"]+=random.uniform(10,20)

            elif anomaly=="Sensor_Error":

                df.loc[idx,"Body_Temperature_C"]+=random.uniform(-0.40,0.40)

            elif anomaly=="Heat_Event":

                df.loc[idx,"THI"]+=random.uniform(3,8)

            elif anomaly=="Fatigue":

                df.loc[idx,"Activity_Index"]*=random.uniform(0.60,0.85)


# ----------------------------------------------------------
# Healthy Cow Natural Variation
# ----------------------------------------------------------

healthy_idx = df[df["Health_Status"] == "Healthy"].sample(
    frac=0.03,
    random_state=SEED
).index

for idx in healthy_idx:

    # Small physiological fluctuation

    df.loc[idx, "Heart_Rate_BPM"] += random.uniform(-3, 3)

    df.loc[idx, "Body_Temperature_C"] += random.uniform(-0.20, 0.20)

    df.loc[idx, "Rumination_Min"] += random.uniform(-20, 20)

    df.loc[idx, "Eating_Hr"] += random.uniform(-0.40, 0.40)

    df.loc[idx, "Drinking_Min"] += random.uniform(-2, 2)

    df.loc[idx, "Walking_km"] += random.uniform(-0.60, 0.60)

    df.loc[idx, "Standing_Hr"] += random.uniform(-0.50, 0.50)

    df.loc[idx, "Lying_Hr"] += random.uniform(-0.50, 0.50)

    df.loc[idx, "Activity_Index"] += random.uniform(-8, 8)


# ----------------------------------------------------------
# Clip to Realistic Range
# ----------------------------------------------------------

df["Heart_Rate_BPM"] = df["Heart_Rate_BPM"].clip(45,130)

df["Body_Temperature_C"] = df["Body_Temperature_C"].clip(37.2,42.5)

df["Rumination_Min"] = df["Rumination_Min"].clip(80,700)

df["Eating_Hr"] = df["Eating_Hr"].clip(0,8)

df["Drinking_Min"] = df["Drinking_Min"].clip(1,40)

df["Walking_km"] = df["Walking_km"].clip(0,12)

df["Standing_Hr"] = df["Standing_Hr"].clip(0,24)

df["Lying_Hr"] = df["Lying_Hr"].clip(0,24)

df["Activity_Index"] = df["Activity_Index"].clip(0,100)

df["THI"] = df["THI"].clip(45,95)


# ----------------------------------------------------------
# Disease Label Encoding
# ----------------------------------------------------------

label_map = {

    "Healthy":0,

    "At_Risk":1,

    "Diseased":2

}

df["Disease_Label"] = df["Health_Status"].map(label_map).astype(int)


# ----------------------------------------------------------
# Arrange Columns
# ----------------------------------------------------------

columns=[

"Sample_ID",
"Cow_ID",
"Day",
"Hour",

"Heart_Rate_BPM",
"Body_Temperature_C",

"Rumination_Min",
"Eating_Hr",
"Drinking_Min",

"Walking_km",
"Standing_Hr",
"Lying_Hr",
"Activity_Index",

"Ambient_Temperature_C",
"Relative_Humidity",
"THI",

"Health_Status",
"Disease_Name",
"Disease_Label"

]

df=df[columns]


# ----------------------------------------------------------
# Save Dataset
# ----------------------------------------------------------

os.makedirs("generated",exist_ok=True)

df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("="*60)
print("Disease Injection Completed Successfully")
print("="*60)

print()

print("Dataset Shape")
print(df.shape)

print()

print("Health Status Distribution")
print(df["Health_Status"].value_counts())

print()

print("Disease Distribution")
print(df["Disease_Name"].value_counts())

print()

print("Disease Label Distribution")
print(df["Disease_Label"].value_counts())

print()

print(df.describe())

print()

print("Saved :",OUTPUT_FILE)

print("="*60)