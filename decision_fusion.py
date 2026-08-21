import os
import pandas as pd

VISION = "vision/output/behavior_summary.csv"
SENSOR = "generated/sensor_data.csv"
RISK = "generated/risk_scores.csv"

OUTPUT = "generated/multimodal_results.csv"

print("="*60)
print("Decision Fusion Engine")
print("="*60)

vision = pd.read_csv(VISION)
sensor = pd.read_csv(SENSOR)
risk = pd.read_csv(RISK)

# ----------------------------
# Merge
# ----------------------------

df = vision.merge(sensor,on="Cow_ID")
df = df.merge(risk[["Cow_ID","Risk_Score","Risk_Level"]],on="Cow_ID")

# ----------------------------
# Decision Fusion
# ----------------------------

def final_decision(row):

    score = row["Risk_Score"]
    behavior = str(row["Dominant_Behavior"]).lower()

    if score >= 70:
        return "Diseased"

    elif score >= 40:

        if behavior in ["lying","standing","walking"]:
            return "At Risk"

        elif behavior=="rumination":
            return "Healthy"

        else:
            return "At Risk"

    else:

        return "Healthy"

df["Final_Status"] = df.apply(final_decision,axis=1)

os.makedirs("generated",exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)

print()
print(df[
[
" Cow_ID".strip(),
"Dominant_Behavior",
"Risk_Score",
"Risk_Level",
"Final_Status"
]].head())

print()

print("Status Distribution")

print(df["Final_Status"].value_counts())

print()

print("Saved")

print(OUTPUT)

print("="*60)
print("Decision Fusion Completed")
print("="*60)