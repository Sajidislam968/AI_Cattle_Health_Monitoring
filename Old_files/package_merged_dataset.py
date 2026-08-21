"""Create a ZIP file of the merged YOLO dataset for Colab or Kaggle upload.

Run locally:
    python3 package_merged_dataset.py
"""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "datasets" / "merged_behavior"
ARCHIVE_BASE = PROJECT_ROOT / "merged_behavior_yolo11"


def main():
    if not (SOURCE_DIR / "data.yaml").exists():
        raise FileNotFoundError(f"Merged dataset not found: {SOURCE_DIR}")

    archive = shutil.make_archive(
        base_name=str(ARCHIVE_BASE),
        format="zip",
        root_dir=str(SOURCE_DIR.parent),
        base_dir=SOURCE_DIR.name,
    )
    size_mb = Path(archive).stat().st_size / 1024**2
    print(f"Dataset archive created: {archive}")
    print(f"Size: {size_mb:.1f} MB")
    print("Upload this ZIP to Google Colab or Kaggle.")


if __name__ == "__main__":
    main()
