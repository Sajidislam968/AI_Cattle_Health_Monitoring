# ==========================================================
# load_yolo_model.py
#
# Phase-7
# Vision AI
#
# Load Pretrained YOLO11 Model
#
# Output
#   vision/models/yolo11m.pt
#
# ==========================================================

import os

from ultralytics import YOLO

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

model = YOLO("yolo11m.pt")

MODEL_DIR = "vision/models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "yolo11m.pt"
)

print("=" * 60)
print("Loading Pretrained YOLO Model")
print("=" * 60)
# ----------------------------------------------------------
# Download Model (if needed)
# ----------------------------------------------------------

if os.path.exists(MODEL_PATH):

    print()

    print("Model already exists.")

    model = YOLO(MODEL_PATH)

else:

    print()

    print("Downloading YOLO11 pretrained model...")

    model = YOLO(model)

    model.save(MODEL_PATH)

print()

print("Model Loaded Successfully")

print("Model :", model)

# ----------------------------------------------------------
# Model Information
# ----------------------------------------------------------

print()
print("=" * 60)
print("YOLO Model Information")
print("=" * 60)

print(f"Task            : {model.task}")

print(f"Model Name      : {model}")

print(f"Model Path      : {MODEL_PATH}")

print(f"Device          : Auto")

print(f"Classes         : {len(model.names)}")

print()

print("First 15 Classes")

print("-" * 40)

for class_id, class_name in list(model.names.items())[:15]:

    print(f"{class_id:02d} -> {class_name}")

# ----------------------------------------------------------
# Verify Cow Class
# ----------------------------------------------------------

print()

print("=" * 60)

print("Searching Cow Class")

print("=" * 60)

cow_id = None

for idx, name in model.names.items():

    if name.lower() == "cow":

        cow_id = idx

        break

if cow_id is None:

    print()

    print("Cow class was NOT found.")

else:

    print()

    print("Cow class detected successfully.")

    print(f"Class ID : {cow_id}")

print()

print("=" * 60)

print("YOLO Model Ready")

print("=" * 60)

print()

print("Next Step")

print(" -> cow_detection.py")