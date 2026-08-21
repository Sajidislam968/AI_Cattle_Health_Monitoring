"""Train the merged YOLO dataset on a Colab or Kaggle GPU.

Upload this script and merged_behavior_yolo11.zip to the cloud session, then:
    python train_yolo11_cloud.py --dataset-zip /content/merged_behavior_yolo11.zip

Kaggle example:
    python train_yolo11_cloud.py --dataset-zip /kaggle/input/YOUR-DATASET/merged_behavior_yolo11.zip
"""

import argparse
from pathlib import Path
import shutil
import zipfile

import torch
from ultralytics import YOLO


CLASS_NAMES = ["standing", "walking", "eating", "lying", "rumination"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", required=True, help="Path to merged_behavior_yolo11.zip")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolo11m.pt", help="Use yolo11n.pt if GPU memory is limited")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 chooses batch size automatically")
    parser.add_argument("--workdir", default="/content/yolo_behavior_training")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_zip = Path(args.dataset_zip).resolve()
    workdir = Path(args.workdir).resolve()
    dataset_parent = workdir / "datasets"
    dataset_dir = dataset_parent / "merged_behavior"
    results_dir = workdir / "results"

    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {dataset_zip}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. In Colab enable Runtime > Change runtime type > GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Extracting dataset...")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as archive:
        archive.extractall(dataset_parent)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml missing after extraction: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=str(results_dir),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=20,
        seed=42,
        plots=True,
    )

    result_zip = shutil.make_archive(str(workdir / "yolo11_behavior_results"), "zip", results_dir)
    print(f"\nTraining complete. Download this file: {result_zip}")
    print(f"Best model: {results_dir / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
