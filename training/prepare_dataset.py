#!/usr/bin/env python3
"""Build a YOLO detection dataset from the unlabeled thumbs-up images.

Two data sources are handled:
  * silhouette images (240x215) -> bright-pixel segmentation gives the hand box
  * real photographs (640x640)  -> MediaPipe hand landmarks give hand boxes,
    landmarks + canned gesture recognizer decide the class

Classes:
  0 = thumbs_up
  1 = other_gesture
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "thumbs up dataset"
OUT = ROOT / "dataset"
MODEL = ROOT / "models" / "gesture_recognizer.task"

FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))


def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_features(lm) -> tuple[float, float, float]:
    """Return (finger_curl_ratio, thumb_extension_ratio, thumb_above_palm)."""
    wrist = lm[0]
    ratios = [
        dist(lm[tip], wrist) / max(dist(lm[pip], wrist), 1e-6)
        for tip, pip in FINGERS
    ]
    thumb_ext = dist(lm[4], lm[5]) / max(dist(lm[2], lm[5]), 1e-6)
    thumb_above = lm[2].y - lm[4].y
    return float(np.mean(ratios)), float(thumb_ext), float(thumb_above)


def classify_hand(gesture, lm) -> int:
    """0 = thumbs up, 1 = other gesture."""
    name = gesture[0].category_name if gesture else "None"
    if name == "Thumb_Up":
        return 0
    if name in {"Thumb_Down", "Victory"}:
        return 1

    ratio, thumb_ext, thumb_above = hand_features(lm)
    if ratio < 0.95 and thumb_ext > 1.3 and thumb_above > 0.02:
        return 0
    return 1


def landmarks_to_box(lm, width: int, height: int, pad: float = 0.18):
    xs = [p.x * width for p in lm]
    ys = [p.y * height for p in lm]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    ]


def to_yolo_line(cls: int, box, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    xc = (x1 + x2) / 2 / width
    yc = (y1 + y2) / 2 / height
    return f"{cls} {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}"


def silhouette_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 30)
    if len(xs) < 50:
        return []
    h, w = gray.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x, pad_y = (x2 - x1) * 0.03, (y2 - y1) * 0.03
    return [[max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--photo-repeat", type=int, default=2, help="oversample real photos in train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sources", choices=("all", "photos"), default="all")
    args = parser.parse_args()
    out = args.out
    if not out.is_absolute():
        out = ROOT / out

    files = sorted(SRC.glob("*.jpg"))
    silhouettes = [p for p in files if ".rf." not in p.name]
    photos = [p for p in files if ".rf." in p.name]
    if args.sources == "photos":
        files = photos
    print(f"found {len(silhouettes)} silhouettes and {len(photos)} photos ({args.sources})")

    base = mpp.BaseOptions(model_asset_path=str(MODEL))
    options = vision.GestureRecognizerOptions(
        base_options=base, running_mode=vision.RunningMode.IMAGE, num_hands=4
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    records = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels: list[tuple[int, list[float]]] = []

        if ".rf." not in path.name:
            for box in silhouette_boxes(image):
                labels.append((0, box))
            source = "silhouette"
        else:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            result = recognizer.recognize(mp_image)
            for lm, gesture in zip(result.hand_landmarks, result.gestures):
                cls = classify_hand(gesture, lm)
                labels.append((cls, landmarks_to_box(lm, w, h)))
            source = "photo"

        records.append(
            {
                "path": path,
                "source": source,
                "labels": labels,
                "has_thumbs_up": any(c == 0 for c, _ in labels),
            }
        )
    recognizer.close()

    rng = random.Random(args.seed)

    def split(items, val_frac):
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        return items[n_val:], items[:n_val]

    # stratify photos so thumbs-up examples appear in both splits
    sil_records = [r for r in records if r["source"] == "silhouette"]
    sil_train, sil_val = split(sil_records, args.val_fraction)
    pos_photos = [r for r in records if r["source"] == "photo" and r["has_thumbs_up"]]
    neg_photos = [r for r in records if r["source"] == "photo" and not r["has_thumbs_up"]]
    pos_train, pos_val = split(pos_photos, 0.2)
    neg_train, neg_val = split(neg_photos, 0.2)

    train = sil_train + pos_train + neg_train
    val = sil_val + pos_val + neg_val

    if out.exists():
        shutil.rmtree(out)
    for split_name in ("train", "val"):
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(Counter)
    for split_name, items, repeat in (
        ("train", train, args.photo_repeat),
        ("val", val, 1),
    ):
        for rec in items:
            image = cv2.imread(str(rec["path"]))
            h, w = image.shape[:2]
            lines = [
                line
                for cls, box in rec["labels"]
                if (line := to_yolo_line(cls, box, w, h)) is not None
            ]
            if not lines and rec["source"] == "silhouette":
                continue
            copies = repeat if rec["source"] == "photo" and split_name == "train" else 1
            for copy_idx in range(copies):
                stem = rec["path"].stem + (f"_r{copy_idx}" if copy_idx else "")
                image_name = f"{stem}.jpg"
                shutil.copy2(rec["path"], out / "images" / split_name / image_name)
                with open(out / "labels" / split_name / f"{stem}.txt", "w") as fh:
                    fh.write("\n".join(lines))
                counts[split_name][rec["source"]] += 1
                for cls, _ in rec["labels"]:
                    counts[split_name][f"class_{cls}"] += 1

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: thumbs_up\n  1: other_gesture\n"
    )
    (out / "stats.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"dataset written to {out}")


if __name__ == "__main__":
    main()
