# ==========================================================
# recommendation_engine.py
#
# Phase-8
# XAI-Based Recommendation Engine
#
# Input:
# generated/shap_explanations.csv
#
# Output:
# generated/final_ai_health_dataset.csv
#
# Purpose:
# Generate recommendations using:
#
# 1. Predicted Health Status
# 2. Risk Score
# 3. Risk Level
# 4. Critical Alert
# 5. SHAP Dominant Factor
# 6. SHAP Secondary Factor
# 7. Physiological Measurements
# 8. Veterinary/Literature Based Thresholds
#
# ==========================================================

import os
import pandas as pd


# ==========================================================
# File Paths
# ==========================================================

INPUT_FILE = (
    "generated/shap_explanations.csv"
)

OUTPUT_FILE = (
    "generated/final_ai_health_dataset.csv"
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading SHAP XAI Dataset")
print("=" * 60)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"{INPUT_FILE} not found."
    )


df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print()
print("Dataset Shape :", df.shape)

print()
print("Columns Loaded :")

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


# ==========================================================
# Recommendation Function
# ==========================================================

def generate_recommendation(row):

    status = row["Predicted_Status"]

    risk_level = row.get(
        "Risk_Level",
        "Unknown"
    )

    dominant = row.get(
        "Dominant_Factor",
        "Unknown"
    )

    secondary = row.get(
        "Secondary_Factor",
        "Unknown"
    )

    recommendations = []


    # ======================================================
    # Critical Alert
    # ======================================================

    # ======================================================
    # Heart Rate
    # ======================================================

    heart_rate = row.get(
        "Heart_Rate_BPM",
        None
    )

    if pd.notna(heart_rate):

        if heart_rate > 84:

            recommendations.append(
                "Heart rate is above the normal range. "
                "Monitor stress, pain and cardiovascular "
                "condition and consider veterinary assessment."
            )

        elif heart_rate < 48:

            recommendations.append(
                "Heart rate is below the normal range. "
                "Check for weakness, reduced activity or "
                "possible cardiovascular/metabolic problems."
            )


    # ======================================================
    # Body Temperature
    # ======================================================

    temperature = row.get(
        "Body_Temperature_C",
        None
    )

    if pd.notna(temperature):

        if temperature > 39.5:

            recommendations.append(
                "Elevated body temperature detected. "
                "Monitor for fever or infection and "
                "consider veterinary evaluation."
            )

        elif temperature < 38.5:

            recommendations.append(
                "Body temperature is below the normal range. "
                "Monitor for weakness or metabolic problems."
            )


    # ======================================================
    # Rumination
    # ======================================================

    rumination = row.get(
        "Rumination_Min",
        None
    )

    if pd.notna(rumination):

        if rumination < 400:

            recommendations.append(
                "Reduced rumination detected. "
                "Monitor digestive health, feeding behavior "
                "and possible illness."
            )


    # ======================================================
    # Feeding
    # ======================================================

    eating = row.get(
        "Eating_Hr",
        None
    )

    if pd.notna(eating):

        if eating < 4:

            recommendations.append(
                "Reduced eating duration detected. "
                "Check feed quality, appetite and general health."
            )


    # ======================================================
    # Drinking
    # ======================================================

    drinking = row.get(
        "Drinking_Min",
        None
    )

    if pd.notna(drinking):

        if drinking < 8:

            recommendations.append(
                "Reduced drinking activity detected. "
                "Ensure clean water availability and monitor hydration."
            )


    # ======================================================
    # Walking Activity
    # ======================================================

    walking = row.get(
        "Walking_km",
        None
    )

    if pd.notna(walking):

        if walking < 3:

            recommendations.append(
                "Reduced walking activity detected. "
                "Observe mobility and check for possible lameness "
                "or discomfort."
            )


    # ======================================================
    # Standing
    # ======================================================

    standing = row.get(
        "Standing_Hr",
        None
    )

    if pd.notna(standing):

        if standing > 12:

            recommendations.append(
                "Increased standing duration detected. "
                "Observe the cow for discomfort or mobility problems."
            )


    # ======================================================
    # Lying
    # ======================================================

    lying = row.get(
        "Lying_Hr",
        None
    )

    if pd.notna(lying):

        if lying > 14:

            recommendations.append(
                "Excessive lying duration detected. "
                "Check for pain, weakness or illness."
            )


    # ======================================================
    # Activity Index
    # ======================================================

    activity = row.get(
        "Activity_Index",
        None
    )

    if pd.notna(activity):

        if activity < 60:

            recommendations.append(
                "Low activity level detected. "
                "Increase observation and compare with the cow's "
                "normal activity baseline."
            )


    # ======================================================
    # Heat Stress / THI
    # ======================================================

    thi = row.get(
        "THI",
        None
    )

    if pd.notna(thi):

        if thi >= 89:

            recommendations.append(
                "Severe heat stress condition detected. "
                "Provide immediate cooling, shade, ventilation "
                "and sufficient drinking water."
            )

        elif thi >= 79:

            recommendations.append(
                "Moderate heat stress detected. "
                "Provide shade, ventilation, cooling and "
                "additional drinking water."
            )

        elif thi >= 72:

            recommendations.append(
                "Mild heat stress detected. "
                "Monitor the cow and environmental conditions closely."
            )


    # ======================================================
    # SHAP-Based Recommendation
    # ======================================================

    if status == "Healthy":

        recommendations.append(
            f"SHAP analysis identified {dominant} as the "
            f"dominant contributing factor and {secondary} "
            f"as the secondary factor. Continue routine monitoring."
        )


    elif status == "At_Risk":

        recommendations.append(
            f"SHAP analysis identified {dominant} as the "
            f"dominant factor and {secondary} as the secondary "
            f"factor. Increase monitoring frequency and compare "
            f"these measurements with the cow's individual baseline."
        )

        recommendations.append(
            "If abnormal patterns persist or worsen, "
            "perform veterinary assessment."
        )


    elif status == "Diseased":

        recommendations.append(
            f"SHAP analysis identified {dominant} as the strongest "
            f"contributing factor and {secondary} as the secondary "
            f"factor."
        )

        recommendations.append(
            "Immediate animal health assessment is recommended. "
            "Use veterinary examination for clinical confirmation "
            "before treatment decisions."
        )


    # ======================================================
    # Risk Level Specific Recommendation
    # ======================================================

    if risk_level == "Critical":

        recommendations.append(
            "Risk level is Critical. Prioritize this animal "
            "for immediate inspection."
        )

    elif risk_level == "High":

        recommendations.append(
            "Risk level is High. Increase monitoring frequency "
            "and investigate persistent abnormalities."
        )

    elif risk_level == "Moderate":

        recommendations.append(
            "Risk level is Moderate. Continue closer monitoring "
            "and compare measurements against the individual baseline."
        )


    # ======================================================
    # Remove Duplicate Recommendations
    # ======================================================

    unique_recommendations = []

    for recommendation in recommendations:

        if recommendation not in unique_recommendations:

            unique_recommendations.append(
                recommendation
            )


    # ======================================================
    # Healthy Fallback
    # ======================================================

    if len(unique_recommendations) == 0:

        unique_recommendations.append(
            "No abnormal physiological condition detected. "
            "Continue routine monitoring."
        )


    return " ".join(
        unique_recommendations
    )


# ==========================================================
# Generate Recommendations
# ==========================================================

print()
print("=" * 60)
print("Generating XAI-Based Recommendations")
print("=" * 60)

recommendations = []

for _, row in df.iterrows():

    recommendation = generate_recommendation(
        row
    )

    recommendations.append(
        recommendation
    )


# ==========================================================
# Add Recommendation
# ==========================================================

df = df.copy()

df.loc[:, "Recommendation"] = (
    recommendations
)


# ==========================================================
# Recommendation Category
# ==========================================================

def recommendation_category(status):

    if status == "Healthy":

        return "Routine Monitoring"

    elif status == "At_Risk":

        return "Increased Monitoring"

    elif status == "Diseased":

        return "Veterinary Assessment"

    return "Observation"


df.loc[:, "Recommendation_Category"] = (
    df["Predicted_Status"]
    .apply(recommendation_category)
)

def priority_level(row):

    status = row["Predicted_Status"]

    if status == "Diseased":
        return "High"

    elif status == "At_Risk":
        return "Medium"

    else:
        return "Low"

# ----------------------------------------------------------
# Recommendation Priority
# ----------------------------------------------------------

df["Recommendation_Priority"] = df.apply(
    priority_level,
    axis=1
)
# ==========================================================
# Save Dataset
# ==========================================================

os.makedirs(
    "generated",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# Statistics
# ==========================================================

print()
print(
    "Recommendations Generated Successfully"
)

print()

print(
    "Recommendation Category Distribution"
)

print(
    df[
        "Recommendation_Category"
    ].value_counts()
)

print()

print(
    "Recommendation Priority Distribution"
)

print(
    df[
        "Recommendation_Priority"
    ].value_counts()
)

print()

print(
    "Prediction Distribution"
)

print(
    df[
        "Predicted_Status"
    ].value_counts()
)

print()


# ==========================================================
# Preview
# ==========================================================

print()
print("=" * 60)
print("Recommendation Dataset Preview")
print("=" * 60)

preview_columns = [
    "Predicted_Status",
    "Confidence_Score",
    "Dominant_Factor",
    "Secondary_Factor",
    "Recommendation_Category",
    "Recommendation_Priority",
    "Recommendation"
]

available_preview_columns = [
    column
    for column in preview_columns
    if column in df.columns
]

print(
    df[
        available_preview_columns
    ].head(5).to_string(
        index=False
    )
)


# ==========================================================
# Final
# ==========================================================

print()
print("=" * 60)
print("RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Total Samples :", len(df))

print()
print("Output File   :")
print(
    OUTPUT_FILE
)

print()
print(
    "The final AI health monitoring dataset "
    "is ready."
)