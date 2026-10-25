#!/usr/bin/env python3
"""Audit apparent false thumbs-up detections in the negative crop set."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))
import app as pipeline  # noqa: E402

SOURCE = ROOT / "dataset_cls_clean" / "val" / "other_gesture"
OUT = ROOT / "graphs"


def main() -> None:
    paths = sorted(SOURCE.glob("*.jpg"))
    rows = []
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.flat:
        ax.axis("off")
    for index, (path, ax) in enumerate(zip(paths, axes.flat)):
        image = cv2.imread(str(path))
        detections = pipeline.run_pipeline(image, pipeline.DEFAULT_CONFIDENCE)
        rows.append({"index": index, "image": path.relative_to(ROOT).as_posix(),
                     "reported_thumbs_up": bool(detections),
                     "top_confidence": max((d["confidence"] for d in detections), default=0.0)})
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                       edgecolor="#ef4444", linewidth=2))
        ax.set_title(f"{index}: {'THUMBS UP ' + str(rows[-1]['top_confidence']) if detections else 'No detection'}",
                     color="#b91c1c" if detections else "#334155", fontsize=9)
        ax.axis("off")
    fig.suptitle("Other-gesture folder audit: red boxes are app outputs; inspect labels visually", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "other_gesture_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    with (OUT / "other_gesture_audit.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"reported thumbs up on {sum(row['reported_thumbs_up'] for row in rows)}/{len(rows)} images")


if __name__ == "__main__":
    main()
