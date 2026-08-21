# ==========================================================
# tracking_engine.py
#
# Phase-7
# Vision AI
#
# Multi-Cow Tracking using YOLO11 + ByteTrack
#
# Input
#    vision/input/cow.mp4
#
# Output
#    vision/output/tracked_video.mp4
#    vision/output/tracking_results.csv
#    vision/output/cow_statistics.csv
#
# ==========================================================

import os
import cv2
import random
import pandas as pd

from collections import defaultdict
from ultralytics import YOLO

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_PATH = "vision/models/yolo11m.pt"

INPUT_VIDEO = "vision/input/cow.mp4"

OUTPUT_DIR = "vision/output"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "tracked_video.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "tracking_results.csv"
)

OUTPUT_STATISTICS = os.path.join(
    OUTPUT_DIR,
    "cow_statistics.csv"
)

CONFIDENCE = 0.40

COW_CLASS = 19

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

print("=" * 60)
print("Cow Tracking Engine")
print("=" * 60)

# ----------------------------------------------------------
# Load YOLO Model
# ----------------------------------------------------------

print()
print("Loading YOLO11 Model...")

model = YOLO(MODEL_PATH)

print("YOLO Model Loaded Successfully")

# ----------------------------------------------------------
# Open Input Video
# ----------------------------------------------------------

print()
print("Opening Input Video...")

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():

    raise FileNotFoundError(
        f"Unable to open : {INPUT_VIDEO}"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Video Loaded Successfully")

print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")

# ----------------------------------------------------------
# Create Video Writer
# ----------------------------------------------------------

writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

    cv2.VideoWriter_fourcc(*"mp4v"),

    fps,

    (width, height),

    True
)

if not writer.isOpened():

    raise RuntimeError(
        "Unable to create output video."
    )

print()
print("Video Writer Ready")

# ----------------------------------------------------------
# Tracking Memory
# ----------------------------------------------------------

trajectory = defaultdict(list)

cow_colors = {}

tracking_records = []

frame_number = 0

# ----------------------------------------------------------
# Color Generator
# ----------------------------------------------------------

def get_color(track_id):

    if track_id not in cow_colors:

        random.seed(track_id)

        cow_colors[track_id] = (

            random.randint(40,255),

            random.randint(40,255),

            random.randint(40,255)

        )

    return cow_colors[track_id]

# ----------------------------------------------------------
# Helper
# ----------------------------------------------------------

def draw_text(

    image,

    text,

    x,

    y,

    color=(0,255,255)

):

    cv2.putText(

        image,

        text,

        (x,y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        color,

        2

    )

print()
print("="*60)
print("Initialization Completed")
print("="*60)

print()

print("Next Step")
print("-> Frame Processing")

# ----------------------------------------------------------
# Start Tracking
# ----------------------------------------------------------

print()
print("=" * 60)
print("Tracking Started...")
print("=" * 60)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------------

    results = model.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        classes=[COW_CLASS],

        conf=CONFIDENCE,

        verbose=False

    )

    result = results[0]

    active_ids = set()

    # --------------------------------------------
    # Skip Empty Detection
    # --------------------------------------------

    if result.boxes is None or len(result.boxes) == 0:

        draw_text(
            frame,
            f"Frame : {frame_number}",
            20,
            35,
            (0,255,255)
        )

        draw_text(
            frame,
            "Active Cows : 0",
            20,
            70,
            (0,255,255)
        )

        writer.write(frame)

        continue

    # --------------------------------------------
    # Process Every Cow
    # --------------------------------------------

    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id != COW_CLASS:
            continue

        confidence = float(box.conf[0])

        # --------------------------
        # Tracking ID
        # --------------------------

        if box.id is None:

            track_id = -1

        else:

            track_id = int(box.id.item())

        active_ids.add(track_id)

        # --------------------------
        # Bounding Box
        # --------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        color = get_color(track_id)

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            2

        )

        label = f"Cow {track_id} ({confidence:.2f})"

        cv2.putText(

            frame,

            label,

            (x1, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )

        # --------------------------
        # Center Point
        # --------------------------

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        trajectory[track_id].append(

            (center_x, center_y)

        )

        if len(trajectory[track_id]) > 40:

            trajectory[track_id].pop(0)

        # --------------------------
        # Draw Trajectory
        # --------------------------

        for i in range(

            1,

            len(trajectory[track_id])

        ):

            cv2.line(

                frame,

                trajectory[track_id][i - 1],

                trajectory[track_id][i],

                color,

                2

            )

        # --------------------------
        # Save CSV Row
        # --------------------------

        tracking_records.append({

            "Frame": frame_number,

            "Cow_ID": track_id,

            "Confidence": confidence,

            "X1": x1,

            "Y1": y1,

            "X2": x2,

            "Y2": y2,

            "Width": x2 - x1,

            "Height": y2 - y1,

            "Center_X": center_x,

            "Center_Y": center_y

        })

    # --------------------------------------------
    # Frame Information
    # --------------------------------------------

    draw_text(

        frame,

        f"Frame : {frame_number}",

        20,

        35,

        (0,255,255)

    )

    draw_text(

        frame,

        f"Active Cows : {len(active_ids)}",

        20,

        70,

        (0,255,255)

    )

    draw_text(

        frame,

        f"Tracked IDs : {len(cow_colors)}",

        20,

        105,

        (0,255,255)

    )

    writer.write(frame)

# ----------------------------------------------------------
# Release Resources
# ----------------------------------------------------------

print()
print("=" * 60)
print("Tracking Completed")
print("=" * 60)

cap.release()
writer.release()

try:
    cv2.destroyAllWindows()
except cv2.error:
    pass  # OpenCV compiled without GUI support

# ----------------------------------------------------------
# Save Tracking Results
# ----------------------------------------------------------

tracking_df = pd.DataFrame(tracking_records)

tracking_df.to_csv(

    OUTPUT_CSV,

    index=False

)

print()
print("Tracking Results Saved Successfully")
print(OUTPUT_CSV)

# ----------------------------------------------------------
# Tracking Summary
# ----------------------------------------------------------

if len(tracking_df) > 0:

    unique_cows = tracking_df["Cow_ID"].nunique()

    total_detections = len(tracking_df)

    average_confidence = tracking_df[
        "Confidence"
    ].mean()

    print()
    print("=" * 60)
    print("Tracking Summary")
    print("=" * 60)

    print(f"Frames Processed   : {frame_number}")
    print(f"Total Detections   : {total_detections}")
    print(f"Unique Cow IDs     : {unique_cows}")
    print(f"Average Confidence : {average_confidence:.3f}")

    print()
    print("Detection Per Cow")

    cow_summary = (

        tracking_df

        .groupby("Cow_ID")

        .size()

        .sort_values(ascending=False)

    )

    print(cow_summary)

else:

    print()
    print("No cows detected.")

# ----------------------------------------------------------
# Cow Statistics
# ----------------------------------------------------------

if len(tracking_df) > 0:

    cow_statistics = (

        tracking_df

        .groupby("Cow_ID")

        .agg(

            Total_Frames=("Frame", "count"),

            Average_Confidence=("Confidence", "mean"),

            Avg_Center_X=("Center_X", "mean"),

            Avg_Center_Y=("Center_Y", "mean"),

            Avg_Width=("Width", "mean"),

            Avg_Height=("Height", "mean")

        )

        .reset_index()

    )

    cow_statistics.to_csv(

        OUTPUT_STATISTICS,

        index=False

    )

    print()
    print("Cow Statistics Saved")
    print(OUTPUT_STATISTICS)

# ----------------------------------------------------------
# Final Output
# ----------------------------------------------------------

print()
print("=" * 60)
print("Tracking Engine Finished Successfully")
print("=" * 60)

print()

print(f"Frames Processed : {frame_number}")
print(f"Tracked IDs      : {len(cow_colors)}")
print(f"Trajectories     : {len(trajectory)}")

print()

print("Generated Files")
print("-" * 40)
print("vision/output/tracked_video.mp4")
print("vision/output/tracking_results.csv")
print("vision/output/cow_statistics.csv")

print()

print("Next Step")
print("-> pose_estimation.py")