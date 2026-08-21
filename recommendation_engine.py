import pandas as pd

INPUT = "generated/final_prediction.csv"
OUTPUT = "generated/final_prediction.csv"

df = pd.read_csv(INPUT)


def add(rec, text):
    if text not in rec:
        rec.append(text)


def recommendation(row):

    rec = []

    # ----------------------------------------------------
    # Final Health Status
    # ----------------------------------------------------

    if row["Final_Status"] == "Healthy":
        add(rec, "Continue routine monitoring.")
        add(rec, "Maintain current feeding schedule.")

    elif row["Final_Status"] == "At Risk":
        add(rec, "Increase monitoring frequency.")
        add(rec, "Inspect the animal within the next few hours.")

    elif row["Final_Status"] == "Diseased":
        add(rec, "Isolate the animal immediately.")
        add(rec, "Contact a veterinarian as soon as possible.")

    # ----------------------------------------------------
    # Temperature
    # ----------------------------------------------------

    temp = row["Body_Temperature"]

    if temp >= 40:
        add(rec, "Possible fever detected.")
        add(rec, "Perform clinical examination.")

    elif temp >= 39.5:
        add(rec, "Monitor body temperature every 2 hours.")

    elif temp <= 38.2:
        add(rec, "Possible hypothermia detected.")

    # ----------------------------------------------------
    # Heart Rate
    # ----------------------------------------------------

    hr = row["Heart_Rate"]

    if hr > 84:
        add(rec, "Reduce stress and recheck heart rate.")

    elif hr < 50:
        add(rec, "Check cardiovascular condition.")

    # ----------------------------------------------------
    # THI
    # ----------------------------------------------------

    thi = row["THI"]

    if thi >= 89:
        add(rec, "Severe heat stress.")
        add(rec, "Provide cooling and unlimited clean water.")

    elif thi >= 79:
        add(rec, "Moderate heat stress.")
        add(rec, "Increase ventilation.")

    elif thi >= 72:
        add(rec, "Mild heat stress.")
        add(rec, "Ensure adequate drinking water.")

    # ----------------------------------------------------
    # Humidity
    # ----------------------------------------------------

    hum = row["Humidity"]

    if hum > 85:
        add(rec, "Reduce barn humidity if possible.")

    elif hum < 35:
        add(rec, "Environment is dry. Monitor dehydration.")

    # ----------------------------------------------------
    # Eating
    # ----------------------------------------------------

    eat = row["Eating_Time"]

    if eat < 240:
        add(rec, "Reduced feeding behaviour detected.")

    elif eat > 420:
        add(rec, "Feed intake appears normal.")

    # ----------------------------------------------------
    # Rumination
    # ----------------------------------------------------

    rum = row["Rumination_Time"]

    if rum < 300:
        add(rec, "Low rumination detected. Check digestive health.")

    elif rum > 480:
        add(rec, "Healthy rumination pattern observed.")

    # ----------------------------------------------------
    # Walking
    # ----------------------------------------------------

    walk = row["Walking_Time"]

    if walk < 40:
        add(rec, "Very low movement detected.")

    elif walk > 180:
        add(rec, "High locomotion observed.")

    # ----------------------------------------------------
    # Lying
    # ----------------------------------------------------

    lie = row["Lying_Time"]

    if lie > 900:
        add(rec, "Excessive lying detected.")

    elif lie < 300:
        add(rec, "Rest duration is lower than expected.")

    # ----------------------------------------------------
    # Activity
    # ----------------------------------------------------

    activity = row["Activity_Index"]

    if activity < 0.30:
        add(rec, "Very low activity level.")

    elif activity > 0.85:
        add(rec, "Very high activity level.")

    # ----------------------------------------------------
    # Dominant Behaviour
    # ----------------------------------------------------

    behavior = row["Dominant_Behavior"]

    if behavior == "standing":
        add(rec, "Standing behaviour appears normal.")

    elif behavior == "walking":
        add(rec, "Walking behaviour appears normal.")

    elif behavior == "eating":
        add(rec, "Eating behaviour detected.")

    elif behavior == "lying":
        add(rec, "Observe resting posture.")

    elif behavior == "rumination":
        add(rec, "Rumination behaviour detected.")

    # ----------------------------------------------------
    # Combined Rules
    # ----------------------------------------------------

    if behavior == "lying" and activity < 0.35:
        add(rec, "Check for weakness or lameness.")

    if behavior == "eating" and eat < 240:
        add(rec, "Eating detected but duration is insufficient.")

    if thi >= 79 and temp >= 39.5:
        add(rec, "Heat stress with elevated temperature.")

    if hr > 84 and temp > 39.5:
        add(rec, "Possible infection suspected.")

    if row["Risk_Score"] >= 70:
        add(rec, "Immediate health assessment recommended.")

    elif row["Risk_Score"] >= 40:
        add(rec, "Observe continuously during the day.")

    return " | ".join(rec)


df["Recommendation"] = df.apply(recommendation, axis=1)

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("Recommendation Engine Completed")
print("=" * 60)
print(df[["Cow_ID", "Final_Status", "Recommendation"]].head())