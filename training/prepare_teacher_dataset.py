#!/usr/bin/env python3
"""Relabel the photos with a HaGRID hand-gesture teacher and build a YOLO set.

The first annotation pass only used MediaPipe and missed many hands.  This
script uses a pretrained 12-class gesture detector (trained on HaGRID) to label
every hand, collapsing:
    like              -> class 0 (thumbs_up)
    everything else   -> class 1 (other_gesture)
Silhouette images are still labeled from their bright-pixel mask.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "thumbs up dataset"
TEACHER = ROOT / "models" / "teacher_haGRID_yolo26n.pt"


def silhouette_box(image) -> list[float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 30)
    if len(xs) < 50:
        return None
    h, w = gray.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    px, py = (x2 - x1) * 0.03, (y2 - y1) * 0.03
    return [max(0.0, x1 - px), max(0.0, y1 - py), min(float(w), x2 + px), min(float(h), y2 + py)]


def yolo_line(cls: int, box, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    return (
        f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{bw / width:.6f} {bh / height:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset_teacher")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    teacher = YOLO(str(TEACHER))

    files = sorted(SRC.glob("*.jpg"))
    records = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels: list[tuple[int, list[float]]] = []

        if ".rf." not in path.name:
            box = silhouette_box(image)
            if box:
                labels.append((0, box))
            source = "silhouette"
        else:
            result = teacher.predict(
                image, imgsz=640, conf=args.conf, iou=args.iou, device="cpu", verbose=False
            )[0]
            for box in result.boxes:
                name = teacher.names[int(box.cls)]
                cls = 0 if name == "like" else 1
                labels.append((cls, [float(v) for v in box.xyxy[0]]))
            source = "photo"

        records.append(
            {
                "path": path,
                "source": source,
                "labels": labels,
                "has_thumbs_up": any(c == 0 for c, _ in labels),
            }
        )
    print(f"labeled {len(records)} images")

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    def split(items, frac):
        items = list(items)
        rng.shuffle(items)
        n = max(1, round(len(items) * frac))
        return items[n:], items[:n]

    sil = [r for r in records if r["source"] == "silhouette"]
    pos = [r for r in records if r["source"] == "photo" and r["has_thumbs_up"]]
    neg = [r for r in records if r["source"] == "photo" and not r["has_thumbs_up"]]
    sil_tr, sil_va = split(sil, args.val_fraction)
    pos_tr, pos_va = split(pos, args.val_fraction)
    neg_tr, neg_va = split(neg, args.val_fraction)
    train, val = sil_tr + pos_tr + neg_tr, sil_va + pos_va + neg_va

    counts = defaultdict(Counter)
    for split_name, items in (("train", train), ("val", val)):
        for rec in items:
            image = cv2.imread(str(rec["path"]))
            h, w = image.shape[:2]
            lines = [ln for c, b in rec["labels"] if (ln := yolo_line(c, b, w, h))]
            if not lines and rec["source"] == "silhouette":
                continue
            shutil.copy2(rec["path"], out / "images" / split_name / rec["path"].name)
            (out / "labels" / split_name / f"{rec['path'].stem}.txt").write_text("\n".join(lines))
            counts[split_name][rec["source"]] += 1
            for c, _ in rec["labels"]:
                counts[split_name][f"class_{c}"] += 1

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: thumbs_up\n  1: other_gesture\n"
    )
    (out / "stats.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"dataset written to {out}")


if __name__ == "__main__":
    main()
