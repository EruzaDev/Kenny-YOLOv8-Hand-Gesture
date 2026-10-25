#!/usr/bin/env python3
"""Build multi-margin webcam-like crops from the teacher-labelled photos.

Every teacher box becomes several square crops with different amounts of
context.  The tight crop matches a hand held close to the webcam; the wider
crops teach the model to ignore background hands.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset_teacher"
SIZE = 640


def to_xyxy(row, w, h):
    c, x, y, bw, bh = row
    return c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]


def intersect(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def make_crops(split: str, out_split: str, out: Path, margins) -> Counter:
    counts = Counter()
    for lp in sorted((SRC / "labels" / split).glob("*.txt")):
        imgp = SRC / "images" / split / (lp.stem + ".jpg")
        if ".rf." not in imgp.name:
            continue
        image = cv2.imread(str(imgp))
        if image is None:
            continue
        h, w = image.shape[:2]
        rows = []
        for line in lp.read_text().splitlines():
            if not line.strip():
                continue
            rows.append(to_xyxy(list(map(float, line.split())), w, h))
        if not rows:
            continue

        for idx, (main_cls, box) in enumerate(rows):
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 8 or bh < 8:
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            for margin in margins:
                side = max(bw, bh) * (1 + 2 * margin)
                x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                x2, y2 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                crop = cv2.resize(
                    image[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA
                )
                cw, ch = x2 - x1, y2 - y1
                crop_rect = [x1, y1, x2, y2]

                labels = []
                for cls, other in rows:
                    if other is not box and intersect(
                        crop_rect, other
                    ) < 0.5 * (other[2] - other[0]) * (other[3] - other[1]):
                        continue
                    bx1, by1 = max(0, other[0] - x1), max(0, other[1] - y1)
                    bx2, by2 = min(cw, other[2] - x1), min(ch, other[3] - y1)
                    if bx2 - bx1 < 6 or by2 - by1 < 6:
                        continue
                    nx, ny = (bx1 + bx2) / 2 / cw, (by1 + by2) / 2 / ch
                    nw, nh = (bx2 - bx1) / cw, (by2 - by1) / ch
                    labels.append(f"{int(cls)} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

                if not any(int(l.split()[0]) == int(main_cls) for l in labels):
                    continue
                tag = f"{int(round(margin * 100)):03d}"
                name = f"{imgp.stem}_{idx:02d}_c{int(main_cls)}_m{tag}"
                cv2.imwrite(str(out / "images" / out_split / f"{name}.jpg"), crop)
                (out / "labels" / out_split / f"{name}.txt").write_text("\n".join(labels))
                counts[f"{out_split}_images"] += 1
                for line in labels:
                    counts[f"class_{line.split()[0]}"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dataset_crops_multi")
    parser.add_argument("--margins", type=float, nargs="+", default=[0.6, 0.9, 1.3])
    args = parser.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    counts = make_crops("train", "train", out, args.margins)
    counts += make_crops("val", "val", out, args.margins)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: thumbs_up\n  1: other_gesture\n"
    )
    (out / "stats.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
