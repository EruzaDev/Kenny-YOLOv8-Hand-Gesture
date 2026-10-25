#!/usr/bin/env python3
"""Validate the final model and write an annotated prediction preview."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "thumbs_up_best.pt"
DATA = ROOT / "dataset_photos"
RUNS = ROOT / "models" / "runs"


def tile(images, cols=4, size=(320, 240)):
    tiles = [cv2.resize(im, size) for im in images]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> None:
    model = YOLO(str(MODEL))
    metrics = model.val(
        data=str(DATA / "data.yaml"),
        imgsz=416,
        device="cpu",
        project=str(RUNS),
        name="final_eval",
        exist_ok=True,
        plots=True,
    )
    summary = {
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class_map50_95": metrics.box.maps.tolist(),
        "class_names": metrics.names,
    }
    (RUNS / "final_eval" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    val_images = sorted((DATA / "images" / "val").glob("*.jpg"))
    # Prefer a mix of images that contain thumbs-up labels.
    positive = []
    for image_path in val_images:
        label_path = DATA / "labels" / "val" / f"{image_path.stem}.txt"
        if label_path.exists() and any(
            line.startswith("0 ") for line in label_path.read_text().splitlines()
        ):
            positive.append(image_path)
    selected = (positive + val_images)[:8]
    results = model.predict(
        [str(p) for p in selected], imgsz=416, conf=0.25, device="cpu", verbose=False
    )
    annotated = [res.plot(line_width=2, labels=True, conf=True) for res in results]
    preview = tile(annotated)
    cv2.imwrite(str(ROOT / "models" / "predictions_preview.jpg"), preview)
    print("saved models/predictions_preview.jpg")


if __name__ == "__main__":
    main()
