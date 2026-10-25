#!/usr/bin/env python3
"""Compare original and reviewed classifier on the same reviewed validation crops."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "dataset_cls_reviewed" / "val"
OUT = ROOT / "graphs"
MODELS = {
    "Current": ROOT / "models" / "thumbs_up_classifier.pt",
    "Reviewed candidate": ROOT / "models" / "thumbs_up_classifier_reviewed_candidate.pt",
}


def main() -> None:
    files = [(path, 1) for path in sorted((VAL / "thumbs_up").glob("*.jpg"))]
    files += [(path, 0) for path in sorted((VAL / "no_thumbs_up").glob("*.jpg"))]
    rows = []
    for name, weights in MODELS.items():
        model = YOLO(str(weights))
        index = next(i for i, label in model.names.items() if label == "thumbs_up")
        predictions = []
        for path, truth in files:
            image = cv2.imread(str(path))
            score = float(model.predict(image, imgsz=224, device="cpu", verbose=False)[0].probs.data[index])
            predictions.append((truth, score))
        for threshold in (0.50, 0.80):
            tp = sum(truth == 1 and score >= threshold for truth, score in predictions)
            tn = sum(truth == 0 and score < threshold for truth, score in predictions)
            fp = sum(truth == 0 and score >= threshold for truth, score in predictions)
            fn = sum(truth == 1 and score < threshold for truth, score in predictions)
            rows.append({"model": name, "threshold": threshold, "tp": tp, "tn": tn,
                         "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(files)})
    with (OUT / "classifier_experiment_comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    names = list(MODELS)
    for offset, threshold, color in ((-0.18, 0.50, "#2563eb"), (0.18, 0.80, "#16a34a")):
        values = [next(row["accuracy"] for row in rows if row["model"] == name and row["threshold"] == threshold)
                  for name in names]
        bars = ax.bar([i + offset for i in range(2)], values, width=0.34,
                      label=f"Decision cutoff {threshold:.0%}", color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.axhline(0.90, linestyle="--", color="#dc2626", label="Requested 90%")
    ax.set_xticks(range(2), names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Standalone classifier accuracy")
    ax.set_title("Same 41 reviewed validation crops · candidate remains below 90%", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "classifier_experiment_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
