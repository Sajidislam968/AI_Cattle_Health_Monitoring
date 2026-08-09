# ==========================================================
# preprocessing.py
#
# Phase-5
# Data Preprocessing
#
# Input:
# generated/xai_dataset.csv
#
# Output:
# processed/
#
#   X_train.csv
#   X_test.csv
#   y_train.csv
#   y_test.csv
#   scaler.pkl
#   label_encoder.pkl
#   feature_columns.pkl
#
# ==========================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------
# File Paths
# ----------------------------------------------------------

INPUT_FILE = "generated/xai_dataset.csv"

OUTPUT_DIR = "processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("Loading XAI Dataset...")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print()

print("Dataset Shape :", df.shape)

print()

print(df.head())

# ----------------------------------------------------------
# Remove Unnecessary Columns
# ----------------------------------------------------------

drop_columns = [

    "Sample_ID",
    "Cow_ID",

    "Disease_Name",

    "Alert_Reason",

    "Recommendation",

    "XAI_Explanation",

    "Dominant_Factor",

    "Secondary_Factor",

    "Feature_Contribution",

    "Confidence_Score"

]

df.drop(
    columns=drop_columns,
    inplace=True
)

print()

print("Columns after Cleaning")

print(df.columns.tolist())

# ----------------------------------------------------------
# Encode Categorical Features
# ----------------------------------------------------------

health_encoder = LabelEncoder()

risk_encoder = LabelEncoder()

df["Health_Status"] = health_encoder.fit_transform(
    df["Health_Status"]
)

df["Risk_Level"] = risk_encoder.fit_transform(
    df["Risk_Level"]
)

joblib.dump(
    health_encoder,
    OUTPUT_DIR + "/health_encoder.pkl"
)

joblib.dump(
    risk_encoder,
    OUTPUT_DIR + "/risk_level_encoder.pkl"
)

print()

print("Categorical Encoding Completed")


# ----------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------

feature_columns = [

    # Physiological Features
    "Heart_Rate_BPM",
    "Body_Temperature_C",

    # Behavioral Features
    "Rumination_Min",
    "Eating_Hr",
    "Drinking_Min",
    "Walking_km",
    "Standing_Hr",
    "Lying_Hr",
    "Activity_Index",

    # Environmental Features
    "Ambient_Temperature_C",
    "Relative_Humidity",
    "THI"

]

target_column = "Disease_Label"

X = df[feature_columns]

y = df[target_column]

print()
print("=" * 60)
print("Feature Selection Completed")
print("=" * 60)
print("Number of Features :", len(feature_columns))
print("Target :", target_column)

# ----------------------------------------------------------
# Save Feature List
# ----------------------------------------------------------

joblib.dump(
    feature_columns,
    OUTPUT_DIR + "/feature_columns.pkl"
)

# ----------------------------------------------------------
# Target Selection
# ----------------------------------------------------------

target_column = "Disease_Label"

y = df[target_column]

print()
print("=" * 60)
print("Target Selection Completed")
print("=" * 60)

print("Target Column :", target_column)

print()

print("Target Distribution")
print(y.value_counts().sort_index())

print()

print("Target Percentage (%)")
print((y.value_counts(normalize=True) * 100).round(2))

# ----------------------------------------------------------
# Save Feature List
# ----------------------------------------------------------

feature_columns = list(X.columns)

joblib.dump(
    feature_columns,
    os.path.join(
        OUTPUT_DIR,
        "feature_columns.pkl"
    )
)

print()
print("=" * 60)
print("Feature List Saved Successfully")
print("=" * 60)

print("Total Features :", len(feature_columns))

print()

print("Feature List")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:02d}. {feature}")


# ----------------------------------------------------------
# Train / Test Split
# ----------------------------------------------------------

print()
print("=" * 60)
print("Splitting Dataset...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print()

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print()

print("Training Class Distribution")
print(y_train.value_counts().sort_index())

print()

print("Testing Class Distribution")
print(y_test.value_counts().sort_index())

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------

print()
print("=" * 60)
print("Applying StandardScaler...")
print("=" * 60)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(
        OUTPUT_DIR,
        "scaler.pkl"
    )
)

# ----------------------------------------------------------
# Convert Back to DataFrame
# ----------------------------------------------------------

X_train = pd.DataFrame(
    X_train,
    columns=feature_columns
)

X_test = pd.DataFrame(
    X_test,
    columns=feature_columns
)

y_train = pd.DataFrame(
    y_train,
    columns=[target_column]
)

y_test = pd.DataFrame(
    y_test,
    columns=[target_column]
)

# ----------------------------------------------------------
# Save Processed Dataset
# ----------------------------------------------------------

X_train.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "X_train.csv"
    ),
    index=False
)

X_test.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "X_test.csv"
    ),
    index=False
)

y_train.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "y_train.csv"
    ),
    index=False
)

y_test.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "y_test.csv"
    ),
    index=False
)

print()
print("=" * 60)
print("Preprocessing Completed Successfully")
print("=" * 60)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print()

print("Files Saved Successfully")

print("processed/X_train.csv")
print("processed/X_test.csv")
print("processed/y_train.csv")
print("processed/y_test.csv")
print("processed/scaler.pkl")
print("processed/health_encoder.pkl")
print("processed/risk_level_encoder.pkl")
print("processed/feature_columns.pkl")
