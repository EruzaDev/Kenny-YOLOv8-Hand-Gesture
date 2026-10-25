#!/usr/bin/env python3
"""Train the close-up hand classifier used by the web app.

Prerequisite: ``python training/prepare_crop_dataset.py`` has produced
``dataset_crops``.  The classifier learns thumbs-up vs other gesture from the
centred hand in each crop, which is the same view the web app feeds it.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "dataset_crops"
CLS_DATA = ROOT / "dataset_cls_clean"
RUNS = ROOT / "models" / "runs"


def build_classification_dataset() -> None:
    if not CROPS.exists():
        raise SystemExit("Run training/prepare_crop_dataset.py first.")
    if CLS_DATA.exists():
        shutil.rmtree(CLS_DATA)
    for split in ("train", "val"):
        for name in ("thumbs_up", "other_gesture"):
            (CLS_DATA / split / name).mkdir(parents=True, exist_ok=True)
        for image in (CROPS / "images" / split).glob("*.jpg"):
            match = re.search(r"_c(\d+)$", image.stem)
            if not match:
                continue
            main_class = int(match.group(1))
            rows = []
            label_path = CROPS / "labels" / split / f"{image.stem}.txt"
            for line in label_path.read_text().splitlines():
                if line.strip():
                    _, _, _, w, h = map(float, line.split())
                    rows.append((int(line.split()[0]), w * h))
            same = sum(area for cls, area in rows if cls == main_class)
            other = sum(area for cls, area in rows if cls != main_class)
            if other > 0.15 * max(same, 1e-9):
                continue
            folder = "thumbs_up" if main_class == 0 else "other_gesture"
            shutil.copy2(image, CLS_DATA / split / folder / image.name)
    print(f"classification data written to {CLS_DATA}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    if not args.skip_data:
        build_classification_dataset()

    model = YOLO("yolov8n-cls.pt")
    result = model.train(
        data=str(CLS_DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=2,
        project=str(RUNS),
        name="thumbsup_classifier",
        exist_ok=True,
        plots=True,
        seed=42,
        deterministic=True,
        cos_lr=True,
        patience=20,
        dropout=0.1,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    target = ROOT / "models" / "thumbs_up_classifier.pt"
    shutil.copy2(best, target)
    print(f"classifier saved to {target}")


if __name__ == "__main__":
    main()
