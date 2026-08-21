# ==========================================================
# cow_detection.py
#
# Phase-7
# Vision AI
#
# Cow Detection using Pretrained YOLO11
#
# Input
#   vision/input/
#
# Output
#   vision/output/
#
# ==========================================================

import os
import cv2

from ultralytics import YOLO

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_PATH = "vision/models/yolo11m.pt"

INPUT_DIR = "vision/input"

OUTPUT_DIR = "vision/output"

COW_CLASS = 19

CONFIDENCE = 0.40

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# Supported File Extensions
# ----------------------------------------------------------

IMAGE_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

VIDEO_EXT = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)

print("=" * 60)
print("Cow Detection Engine")
print("=" * 60)

model = YOLO(MODEL_PATH)

print()
print("YOLO Model Loaded Successfully")

# =====================================================
# Collect Images
# =====================================================

files = []

for root, dirs, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        if filename.lower().endswith(
            IMAGE_EXT + VIDEO_EXT
        ):

            files.append(
                os.path.join(root, filename)
            )

print(f"Files Found : {len(files)}")

# ----------------------------------------------------------
# Image Detection
# ----------------------------------------------------------

for input_path in files:

    file_name = os.path.basename(input_path)

    extension = os.path.splitext(file_name)[1].lower()
    # ------------------------------------------
    # Image
    # ------------------------------------------

    if extension in IMAGE_EXT:

        print()

        print("=" * 60)
        print(f"Processing Image : {file_name}")
        print("=" * 60)

        image = cv2.imread(input_path)

        if image is None:

            print("Unable to read image.")
            continue

        results = model.predict(

            source=image,

            conf=CONFIDENCE,

            verbose=False

        )

        result = results[0]

        detection_count = 0

        # --------------------------------------
        # Draw Bounding Boxes
        # --------------------------------------

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            if class_id != COW_CLASS:
                continue

            detection_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )

            label = f"Cow {confidence:.2f}"

            cv2.putText(

                image,

                label,

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2

            )

        # --------------------------------------
        # Save Result
        # --------------------------------------

        output_path = os.path.join(

            OUTPUT_DIR,

            file_name

        )

        cv2.imwrite(

            output_path,

            image

        )

        print(f"Cows Detected : {detection_count}")

        print(f"Saved : {output_path}")

# ----------------------------------------------------------
# Video Detection
# ----------------------------------------------------------

    elif extension in VIDEO_EXT:

        print()

        print("=" * 60)
        print(f"Processing Video : {file_name}")
        print("=" * 60)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            print("Unable to open video.")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(
            OUTPUT_DIR,
            file_name
        )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = 0
        total_detections = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            results = model.predict(

                source=frame,

                conf=CONFIDENCE,

                verbose=False

            )

            result = results[0]

            cows_this_frame = 0

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id != COW_CLASS:
                    continue

                cows_this_frame += 1
                total_detections += 1

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )

                cv2.putText(

                    frame,

                    f"Cow {confidence:.2f}",

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0,255,0),

                    2

                )

            cv2.putText(

                frame,

                f"Cows : {cows_this_frame}",

                (20,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0,0,255),

                2

            )

            writer.write(frame)

        cap.release()
        writer.release()

        print(f"Frames Processed : {frame_count}")
        print(f"Total Cow Detections : {total_detections}")
        print(f"Saved : {output_path}")

# ----------------------------------------------------------
# Completed
# ----------------------------------------------------------

print()
print("=" * 60)
print("Cow Detection Completed Successfully")
print("=" * 60)

print()

print("Output Folder :")
print(OUTPUT_DIR)

print()

print("Next Step")
print("-> tracking_engine.py")