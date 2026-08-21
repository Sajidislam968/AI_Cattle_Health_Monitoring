"""
===========================================================
Merge Multiple YOLO Datasets
===========================================================

Output

datasets/
    merged_behavior/
        train/
            images/
            labels/

        valid/
            images/
            labels/

        test/
            images/
            labels/

        data.yaml
"""

import os
import random
import shutil
from pathlib import Path
import yaml


def to_detection_label(parts):
    """Return a YOLO detection label, converting polygon labels to xywh.

    Roboflow exports some of the source datasets as segmentation polygons
    (class followed by x/y point pairs).  This merged dataset is intended for
    YOLO detection training, so polygons must be represented by their enclosing
    normalized bounding boxes.
    """
    if len(parts) == 5:
        return parts

    # A polygon is: class_id x1 y1 x2 y2 ...; therefore it has an odd number
    # of values and at least three x/y points.
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None

    try:
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    x_values = coordinates[::2]
    y_values = coordinates[1::2]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    return [
        parts[0],
        f"{(x_min + x_max) / 2:.8f}",
        f"{(y_min + y_max) / 2:.8f}",
        f"{x_max - x_min:.8f}",
        f"{y_max - y_min:.8f}",
    ]

# ==========================================================
# Dataset List
# ==========================================================

ROOT = Path("datasets")

DATASETS = {

    "Cow Behavior Tracking.v6i.yolov11": {

        "path": ROOT / "Cow Behavior Tracking.v6i.yolov11",

        "mapping": {
            # Source names: Drinking, Eating, Sitting, Standing
            1: "eating",
            2: "lying",      # Sitting is grouped with lying/resting
            3: "standing"
        }

    },

    "Cow Lie Stand Walk.v2i.yolov11": {

        "path": ROOT / "Cow Lie Stand Walk.v2i.yolov11",

        "mapping": {
            0: "lying",
            1: "standing",
            2: "walking"
        }

    },

    "cow_behavior_detection.v5i.yolov11": {

        "path": ROOT / "cow_behavior_detection.v5i.yolov11",

        "mapping": {
            # Source names: Estrus, grazing, lying, standing
            # Estrus (0) has no equivalent in FINAL_CLASSES, so it is skipped.
            1: "eating",     # Grazing is grouped with eating
            2: "lying",
            3: "standing"
        }

    },

    "cattle dataset behavior.v1i.yolov11": {

        "path": ROOT / "cattle dataset behavior.v1i.yolov11",

        "mapping": {

            # Source names: Falling, drinking water, eating, foraging,
            # lying down, rumination, sitting, stand-Eat-Sit-Fall, standing.
            # Falling (0), drinking water (1), and mixed activity (7) do not
            # belong to the five requested final classes, so they are skipped.
            2: "eating",
            3: "eating",     # Foraging is grouped with eating
            4: "lying",
            5: "rumination",
            6: "lying",      # Sitting is grouped with lying/resting
            8: "standing"

        }

    }

}

# ==========================================================
# Final Classes
# ==========================================================

FINAL_CLASSES = [
    "standing",
    "walking",
    "eating",
    "lying",
    "rumination"
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(FINAL_CLASSES)
}

# ==========================================================
# Output Folder
# ==========================================================

OUTPUT = ROOT / "merged_behavior"
SPLIT_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
RANDOM_SEED = 42

# Rebuild the generated dataset each time.  Without this, labels left over
# from an earlier mapping can be paired with the newly copied images.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(

        parents=True,
        exist_ok=True

    )

    (OUTPUT / split / "labels").mkdir(

        parents=True,
        exist_ok=True

    )

# ==========================================================
# Merge
# ==========================================================

print("="*60)
print("Merging YOLO Datasets")
print("="*60)

records = []
skipped_labels = 0

for dataset_name, info in DATASETS.items():

    dataset_path = info["path"]

    mapping = info["mapping"]

    print(f"\nDataset : {dataset_name}")

    for source_split in ["train", "valid", "test"]:

        img_dir = dataset_path / source_split / "images"
        lbl_dir = dataset_path / source_split / "labels"

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*"))

        print(source_split, len(images))

        for img in images:

            stem = img.stem

            label_file = lbl_dir / f"{stem}.txt"

            new_name = f"{dataset_name.replace(' ','_')}_{img.name}"

            new_lines = []

            if label_file.exists():
                with open(label_file) as f:

                    for line in f:

                        parts = line.strip().split()

                        if len(parts) < 5:
                            continue

                        old_id = int(parts[0])
                        if old_id not in mapping:

                            skipped_labels += 1
                            continue

                        cls_name = mapping[old_id]

                        new_id = CLASS_TO_ID[cls_name]

                        parts = to_detection_label(parts)

                        if parts is None:
                            skipped_labels += 1
                            continue

                        parts[0] = str(new_id)

                        new_lines.append(" ".join(parts))

            records.append({
                "image": img,
                "name": new_name,
                "labels": new_lines,
                "classes": {int(line.split()[0]) for line in new_lines},
            })

# The original Roboflow splits have no rumination examples in valid/test.
# Create a new deterministic, class-aware 80/10/10 split after merging.
# Each image is assigned to the rarest class it contains, then that group is
# split independently. This guarantees rumination is represented in all splits.
class_image_counts = {class_id: 0 for class_id in range(len(FINAL_CLASSES))}
for record in records:
    for class_id in record["classes"]:
        class_image_counts[class_id] += 1

groups = {class_id: [] for class_id in range(len(FINAL_CLASSES))}
groups[None] = []  # images with no retained target annotation
for record in records:
    group = min(record["classes"], key=class_image_counts.get) if record["classes"] else None
    groups[group].append(record)

rng = random.Random(RANDOM_SEED)
split_records = {split: [] for split in SPLIT_RATIOS}
for group in groups.values():
    rng.shuffle(group)
    total = len(group)
    train_end = round(total * SPLIT_RATIOS["train"])
    valid_end = train_end + round(total * SPLIT_RATIOS["valid"])
    split_records["train"].extend(group[:train_end])
    split_records["valid"].extend(group[train_end:valid_end])
    split_records["test"].extend(group[valid_end:])

image_count = 0
label_count = 0
for split, split_group in split_records.items():
    for record in split_group:
        shutil.copy(record["image"], OUTPUT / split / "images" / record["name"])
        image_count += 1

        if record["labels"]:
            dst_label = OUTPUT / split / "labels" / f"{Path(record['name']).stem}.txt"

            with open(dst_label, "w") as f:

                f.write("\n".join(record["labels"]))

            label_count += 1
# ==========================================================
# Save data.yaml
# ==========================================================

yaml_data = {

    "train": "train/images",

    "val": "valid/images",

    "test": "test/images",

    "nc": len(FINAL_CLASSES),

    "names": FINAL_CLASSES

}

with open(OUTPUT / "data.yaml", "w") as f:

    yaml.dump(yaml_data, f, sort_keys=False)

print("\n")
print("="*60)
print("Merge Completed")
print("="*60)

print("Images :", image_count)
print("Labels :", label_count)

print("\nClasses")

for i, c in enumerate(FINAL_CLASSES):

    print(i, c)

print("\nOutput")
print(OUTPUT)

print()
print("="*60)
print("Merge Summary")
print("="*60)

print("Images :", image_count)
print("Labels :", label_count)
print("Skipped Labels :", skipped_labels)

print()

print("Final Classes")

for i, c in enumerate(FINAL_CLASSES):

    print(f"{i} -> {c}")

print()

print("Merged Dataset")

print(OUTPUT)
