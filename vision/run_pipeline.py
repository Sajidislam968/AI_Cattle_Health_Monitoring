"""
============================================================
Vision Pipeline Runner
============================================================

Runs the complete vision pipeline automatically.

Input
------
vision/input/cow.mp4

Outputs
-------
vision/output/tracked_video.mp4
vision/output/tracking_results.csv
vision/output/crops/
vision/output/behavior_predictions.csv
vision/output/behavior_summary.csv
"""

import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = [

    ("Cow Detection", "cow_detection.py"),

    ("Tracking Engine", "tracking_engine.py"),

    ("Crop Cows", "crop_cows.py"),

    ("Behavior Inference", "behavior_inference.py"),

    ("Behavior Summary", "behavior_summary.py"),

]


def run_step(title, script):

    script_path = os.path.join(BASE_DIR, script)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    start = time.time()

    result = subprocess.run(

        ["python3", script_path],

        cwd=os.path.dirname(BASE_DIR)

    )

    elapsed = time.time() - start

    if result.returncode != 0:

        print(f"\n❌ Failed : {title}")
        sys.exit(result.returncode)

    print(f"✅ Completed : {title}")
    print(f"Time : {elapsed:.2f} sec")


def main():

    print("=" * 60)
    print("Vision Pipeline")
    print("=" * 60)

    input_video = os.path.join(

        BASE_DIR,
        "input",
        "cow.mp4"

    )

    if not os.path.exists(input_video):

        raise FileNotFoundError(

            f"Input video not found:\n{input_video}"

        )

    total_start = time.time()

    for title, script in PIPELINE:

        run_step(title, script)

    total_time = time.time() - total_start

    output_file = os.path.join(

        BASE_DIR,

        "output",

        "behavior_summary.csv"

    )

    print("\n" + "=" * 60)

    if os.path.exists(output_file):

        print("Vision Pipeline Completed Successfully")

        print("=" * 60)

        print()

        print("Generated Outputs")

        print("----------------------------------------")

        print("Tracked Video")
        print("vision/output/tracked_video.mp4")

        print()

        print("Tracking CSV")
        print("vision/output/tracking_results.csv")

        print()

        print("Cow Crops")
        print("vision/output/crops/")

        print()

        print("Behavior Prediction")
        print("vision/output/behavior_predictions.csv")

        print()

        print("Behavior Summary")
        print("vision/output/behavior_summary.csv")

        print()

        print(f"Total Time : {total_time:.2f} sec")

    else:

        print("Pipeline finished but final output was not found.")

        sys.exit(1)


if __name__ == "__main__":

    main()
