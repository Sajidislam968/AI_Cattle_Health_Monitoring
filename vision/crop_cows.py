# ==========================================================
# crop_cows.py
#
# Phase-8
#
# Crop Individual Cows
#
# Input
#   vision/input/cow.mp4
#   vision/output/tracking_results.csv
#
# Output
#   vision/crops/cow_x/frame_xxxxx.jpg
#
# ==========================================================

import os
import cv2
import pandas as pd

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

VIDEO_PATH = "vision/input/cow.mp4"

TRACKING_CSV = "vision/output/tracking_results.csv"

OUTPUT_DIR = "vision/output/crops"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 60)
print("Cow Crop Extraction")
print("=" * 60)

# ----------------------------------------------------------
# Load Tracking CSV
# ----------------------------------------------------------

tracking_df = pd.read_csv(TRACKING_CSV)

print()
print("Tracking Records :", len(tracking_df))

print("Unique Cows :", tracking_df["Cow_ID"].nunique())

# ----------------------------------------------------------
# Open Video
# ----------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise FileNotFoundError(
        f"Unable to open video: {VIDEO_PATH}"
    )

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Video Frames :", total_frames)

# ----------------------------------------------------------
# Group by Frame
# ----------------------------------------------------------

frame_groups = tracking_df.groupby("Frame")

saved_images = 0
current_frame = 0
# ----------------------------------------------------------
# Read Video Frame-by-Frame
# ----------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_frame += 1

    # এই frame-এ কোন cow না থাকলে skip
    if current_frame not in frame_groups.groups:
        continue

    detections = frame_groups.get_group(current_frame)

    # ------------------------------------------------------
    # Process Every Cow in Current Frame
    # ------------------------------------------------------

    for row in detections.itertuples(index=False):

        cow_id = int(row.Cow_ID)

        x1 = max(0, int(row.X1))
        y1 = max(0, int(row.Y1))
        x2 = min(frame.shape[1], int(row.X2))
        y2 = min(frame.shape[0], int(row.Y2))

        # Invalid bounding box হলে skip
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop cow
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # --------------------------------------------------
        # Create Cow Folder
        # --------------------------------------------------

        cow_folder = os.path.join(
            OUTPUT_DIR,
            f"cow_{cow_id}"
        )

        os.makedirs(
            cow_folder,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save Crop
        # --------------------------------------------------

        output_name = (
            f"frame_{current_frame:06d}.jpg"
        )

        output_path = os.path.join(
            cow_folder,
            output_name
        )

        cv2.imwrite(
            output_path,
            crop
        )

        saved_images += 1

    # Progress প্রতি 50 frame পর দেখাবে
    if current_frame % 50 == 0:

        print(
            f"Processed Frame : "
            f"{current_frame}/{total_frames}"
        )

# ----------------------------------------------------------
# Release Video
# ----------------------------------------------------------

cap.release()

try:
    cv2.destroyAllWindows()
except cv2.error:
    pass  # OpenCV compiled without GUI support

# ----------------------------------------------------------
# Crop Statistics
# ----------------------------------------------------------

print()
print("=" * 60)
print("Crop Extraction Summary")
print("=" * 60)

print(f"Processed Frames : {current_frame}")
print(f"Saved Crops      : {saved_images}")

# ----------------------------------------------------------
# Per Cow Statistics
# ----------------------------------------------------------

cow_summary = (

    tracking_df

    .groupby("Cow_ID")

    .size()

    .reset_index(name="Total_Crops")

    .sort_values(
        by="Total_Crops",
        ascending=False
    )

)

summary_path = os.path.join(
    OUTPUT_DIR,
    "crop_statistics.csv"
)

cow_summary.to_csv(
    summary_path,
    index=False
)

print()
print("Crop Statistics")

for row in cow_summary.itertuples(index=False):

    print(
        f"Cow {row.Cow_ID:<3} : "
        f"{row.Total_Crops} crops"
    )

print()

print(f"Unique Cows : {len(cow_summary)}")

print()

print("Statistics Saved")
print(summary_path)

print()

print("=" * 60)
print("Crop Extraction Completed Successfully")
print("=" * 60)

print()

print("Generated Folder")

print("------------------------------")

print("vision/crops/")

print()

print("Generated File")

print("------------------------------")

print("vision/crops/crop_statistics.csv")

print()

print("Next Step")

print("-> prepare_behavior_dataset.py")