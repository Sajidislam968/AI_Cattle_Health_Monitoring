import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# =====================================================
# Paths
# =====================================================

MODEL_PATH = "models/yolo11_behavior_results/train/weights/best.pt"

CROP_DIR = "vision/output/crops"

OUTPUT_CSV = "vision/output/behavior_predictions.csv"

CONFIDENCE_THRESHOLD = 0.25

# =====================================================
# Load Model
# =====================================================

print("=" * 60)
print("Behavior Inference")
print("=" * 60)
print()

print("Loading Behavior Model...")
model = YOLO(MODEL_PATH)

print("Model Loaded Successfully")
print()

# =====================================================
# Collect Images
# =====================================================

image_list = []

IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png"
}

for path in Path(CROP_DIR).rglob("*"):

    if path.suffix.lower() in IMAGE_EXT:

        image_list.append(path)

print("Crop Folder :", CROP_DIR)
print("Images Found :", len(image_list))
print()

if len(image_list) == 0:

    raise FileNotFoundError(
        "No crop images found."
    )

# =====================================================
# Inference
# =====================================================

print("=" * 60)
print("Running Behavior Prediction")
print("=" * 60)

results_list = []

for image_path in tqdm(image_list):

    cow_id = image_path.parent.name

    image = cv2.imread(str(image_path))

    result = model.predict(

        source=image,

        conf=CONFIDENCE_THRESHOLD,

        verbose=False

    )[0]

    if len(result.boxes) == 0:

        results_list.append({

            "Cow_ID": cow_id,
            "Image_Name": image_path.name,
            "Image_Path": str(image_path),
            "Behavior": "Unknown",
            "Confidence": 0.0

        })

        continue

    # highest confidence detection

    best_box = result.boxes.conf.argmax()

    cls_id = int(result.boxes.cls[best_box])

    conf = float(result.boxes.conf[best_box])

    behavior = result.names[cls_id]

    results_list.append({

        "Cow_ID": cow_id,
        "Image_Name": image_path.name,
        "Image_Path": str(image_path),
        "Behavior": behavior,
        "Confidence": round(conf,4)

    })

# =====================================================
# Save CSV
# =====================================================

df = pd.DataFrame(results_list)

os.makedirs(
    "vision/output",
    exist_ok=True
)

df.to_csv(

    OUTPUT_CSV,

    index=False

)

print()
print("=" * 60)
print("Behavior Inference Completed")
print("=" * 60)

print()

print("Total Predictions :", len(df))

print()

print("Behavior Distribution")

print(df["Behavior"].value_counts())

print()

print("Saved File")

print(OUTPUT_CSV)

print()

print("Next Step")
print("-> behavior_summary.py")