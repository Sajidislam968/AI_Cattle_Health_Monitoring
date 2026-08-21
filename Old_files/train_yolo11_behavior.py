"""Fine-tune YOLO11 for the merged cattle-behaviour detection dataset.

Run:
    python3 train_yolo11_behavior.py
"""

from pathlib import Path

import torch
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "datasets" / "merged_behavior" / "data.yaml"
BASE_MODEL = PROJECT_ROOT / "yolo11m.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "behavior_yolo11"

# Training settings. Reduce EPOCHS or set MODEL_NAME to "yolo11n.pt" if your
# computer has limited GPU memory.
MODEL_NAME = str(BASE_MODEL) if BASE_MODEL.exists() else "yolo11m.pt"
EPOCHS = 5
IMAGE_SIZE = 640
BATCH_SIZE = -1  # Ultralytics automatically chooses a safe batch size.
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML was not found: {DATA_YAML}. Run merge_datasets.py first."
        )

    print("=" * 60)
    print("YOLO11 Cattle Behaviour Fine-tuning")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Image size: {IMAGE_SIZE} | Batch: {BATCH_SIZE}")

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=20,
        seed=42,
        plots=True,
        save=True,
    )

    best_weights = OUTPUT_DIR / "train" / "weights" / "best.pt"
    print("\nTraining completed.")
    print(f"Best model: {best_weights}")
    print(f"Results: {OUTPUT_DIR / 'train'}")


if __name__ == "__main__":
    main()
