#!/usr/bin/env python3
"""Evaluate the teacher + YOLOv8 classifier web-app pipeline.

Uses the same gating and crop-margin logic as ``webapp/app.py`` and writes
``models/pipeline_metrics.json`` plus an annotated preview.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import app as pipeline  # noqa: E402

PHOTO_IMAGES = ROOT / "dataset_teacher_photo_val" / "images"
PHOTO_LABELS = ROOT / "dataset_teacher_photo_val" / "labels"
WEBCAM_IMAGES = ROOT / "dataset_webcam_bench" / "images"
WEBCAM_LABELS = ROOT / "dataset_webcam_bench" / "labels"
TRAIN_IMAGES = ROOT / "dataset_teacher" / "images" / "train"
TRAIN_LABELS = ROOT / "dataset_teacher" / "labels" / "train"
OUT = ROOT / "models" / "pipeline_metrics.json"
PREVIEW = ROOT / "models" / "pipeline_preview.jpg"


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _load_ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        class_id, x, y, w, h = map(float, line.split())
        if int(class_id) != 0:
            continue
        boxes.append(
            [
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ]
        )
    return boxes


def evaluate(image_dir: Path, label_dir: Path, conf: float = 0.30) -> dict:
    true_positive = []
    false_positive = []
    false_negative = 0
    images = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = _load_ground_truth(label_path, width, height)
        if not truth:
            continue
        images += 1
        detections = pipeline.run_pipeline(image, conf)
        matched = set()
        for det in detections:
            best_iou, best_index = 0.0, -1
            for index, box in enumerate(truth):
                value = _iou(det["box"], box)
                if value > best_iou:
                    best_iou, best_index = value, index
            if best_iou >= 0.5:
                true_positive.append(det["confidence"])
                matched.add(best_index)
            else:
                false_positive.append(det["confidence"])
        false_negative += len(truth) - len(matched)

    return {
        "images_with_thumbs_up": images,
        "ground_truth_thumbs_up": len(true_positive) + false_negative,
        "true_positives": len(true_positive),
        "false_positives": len(false_positive),
        "false_negatives": false_negative,
        "true_positive_confidence_at_least_0_90": sum(
            value >= 0.90 for value in true_positive
        ),
        "true_positive_mean_confidence": (
            float(np.mean(true_positive)) if true_positive else 0.0
        ),
        "true_positive_median_confidence": (
            float(np.median(true_positive)) if true_positive else 0.0
        ),
    }


def evaluate_negatives(image_dir: Path, label_dir: Path) -> dict:
    images = 0
    false_positives = []
    for label_path in sorted(label_dir.glob("*.txt")):
        if ".rf." not in label_path.stem:
            continue
        image_path = image_dir / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        truth = _load_ground_truth(label_path, image.shape[1], image.shape[0])
        if truth:
            continue
        images += 1
        thumbs = [
            det
            for det in pipeline.run_pipeline(image, 0.30)
            if det["class_id"] == 0
        ]
        if thumbs:
            false_positives.append(
                {
                    "image": label_path.stem,
                    "confidence": max(det["confidence"] for det in thumbs),
                }
            )
    return {
        "images_without_thumbs_up": images,
        "images_with_false_thumbs_up": len(false_positives),
        "false_thumbs_up_at_least_0_90": sum(
            item["confidence"] >= 0.90 for item in false_positives
        ),
        "examples": false_positives[:5],
    }


def make_preview(limit: int = 8) -> None:
    tiles = []
    for label_path in sorted(PHOTO_LABELS.glob("*.txt")):
        if len(tiles) >= limit:
            break
        image_path = PHOTO_IMAGES / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        truth = _load_ground_truth(label_path, image.shape[1], image.shape[0])
        if not truth:
            continue
        for det in pipeline.run_pipeline(image, 0.30):
            x1, y1, x2, y2 = (int(value) for value in det["box"])
            color = (0, 200, 0) if det["class_id"] == 0 else (140, 140, 140)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{det['label']} {det['confidence']:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        tiles.append(cv2.resize(image, (320, 320)))
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    cv2.imwrite(str(PREVIEW), np.vstack(rows))


def main() -> None:
    photo = evaluate(PHOTO_IMAGES, PHOTO_LABELS)
    webcam = evaluate(WEBCAM_IMAGES, WEBCAM_LABELS)
    negative_photo_val = evaluate_negatives(PHOTO_IMAGES, PHOTO_LABELS)
    negative_photo_train = evaluate_negatives(TRAIN_IMAGES, TRAIN_LABELS)
    summary = {
        "pipeline": {
            "hand_detector": pipeline.TEACHER_PATH.name,
            "classifier": pipeline.CLASSIFIER_PATH.name,
            "teacher_like_min_conf": pipeline.LIKE_MIN_CONF,
            "fallback_max_hand_conf": pipeline.FALLBACK_MAX_HAND_CONF,
            "fallback_min_box_fraction": pipeline.FALLBACK_MIN_AREA_FRACTION,
            "crop_margins": list(pipeline.CROP_MARGINS),
        },
        "photo_validation": photo,
        "webcam_crop_benchmark": webcam,
        "negative_photo_check": {
            "photo_validation": negative_photo_val,
            "photo_train": negative_photo_train,
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    make_preview()
    print(json.dumps(summary, indent=2))
    print(f"saved {OUT}")
    print(f"saved {PREVIEW}")


if __name__ == "__main__":
    main()
