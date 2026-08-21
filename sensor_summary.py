import pandas as pd

print("=" * 60)
print("Sensor Summary")
print("=" * 60)

df = pd.read_csv("generated/sensor_data.csv")

def health_status(row):

    if row["Body_Temperature"] > 39.5:
        return "High Temperature"

    if row["Heart_Rate"] > 80:
        return "High Heart Rate"

    if row["THI"] > 84:
        return "Heat Stress"

    return "Normal"

df["Sensor_Status"] = df.apply(health_status, axis=1)

df.to_csv("generated/sensor_summary.csv", index=False)

print(df.head())

print("\nSaved:")
print("generated/sensor_summary.csv")