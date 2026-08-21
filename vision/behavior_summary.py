import pandas as pd

INPUT_FILE = "vision/output/behavior_predictions.csv"
OUTPUT_FILE = "vision/output/behavior_summary.csv"

print("=" * 60)
print("Behavior Summary")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

# ---------------------------------------
# Count behavior per cow
# ---------------------------------------

summary = (
    df.groupby("Cow_ID")["Behavior"]
      .value_counts()
      .unstack(fill_value=0)
)

# Ensure all columns exist
behaviors = [
    "standing",
    "walking",
    "eating",
    "lying",
    "rumination",
    "Unknown"
]

for b in behaviors:
    if b not in summary.columns:
        summary[b] = 0

# Keep column order
summary = summary[behaviors]

summary["Total_Frames"] = summary.sum(axis=1)

# Percentages
summary["Standing_%"] = (
    summary["standing"] / summary["Total_Frames"] * 100
).round(2)

summary["Walking_%"] = (
    summary["walking"] / summary["Total_Frames"] * 100
).round(2)

summary["Eating_%"] = (
    summary["eating"] / summary["Total_Frames"] * 100
).round(2)

summary["Lying_%"] = (
    summary["lying"] / summary["Total_Frames"] * 100
).round(2)

summary["Rumination_%"] = (
    summary["rumination"] / summary["Total_Frames"] * 100
).round(2)

summary["Dominant_Behavior"] = (
    summary[
        [
            "standing",
            "walking",
            "eating",
            "lying",
            "rumination"
        ]
    ]
    .idxmax(axis=1)
)

summary.reset_index(inplace=True)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Total Cows :", len(summary))
print()

print(summary.head())

print()
print("Saved :", OUTPUT_FILE)

print()
print("=" * 60)
print("Behavior Summary Completed")
print("=" * 60)

print()
print("Next Step")
print("-> behavior_feature_generator.py")