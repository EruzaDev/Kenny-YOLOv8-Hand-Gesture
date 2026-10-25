#!/usr/bin/env python3
"""Train the thumbs-up detector.

Stage 1 uses the complete dataset (silhouettes + photos) as a shape warm-up.
Stage 2 fine-tunes on real photographs only, which is the domain the web app
sees through a webcam.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs"

COMMON = dict(
    imgsz=416,
    batch=16,
    device="cpu",
    workers=2,
    cache=True,
    exist_ok=True,
    plots=True,
    seed=42,
    deterministic=True,
    cos_lr=True,
    close_mosaic=10,
    fliplr=0.5,
    degrees=8.0,
    translate=0.12,
    scale=0.55,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.45,
    mosaic=1.0,
    mixup=0.1,
)


def train_stage(data: Path, name: str, epochs: int, weights: str | Path) -> Path:
    model = YOLO(str(weights))
    result = model.train(
        data=str(data / "data.yaml"),
        epochs=epochs,
        name=name,
        project=str(RUNS),
        patience=max(10, epochs // 3),
        **COMMON,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"[{name}] best weights: {best}")
    return best


def photo_val_score(weights: Path) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(ROOT / "dataset_photos" / "data.yaml"),
        imgsz=416,
        device="cpu",
        plots=False,
        verbose=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-epochs", type=int, default=25)
    parser.add_argument("--stage2-epochs", type=int, default=55)
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    weights = ROOT / "yolov8n.pt"
    if not weights.exists():
        weights = "yolov8n.pt"

    candidates: list[tuple[str, Path]] = []
    starter = Path(weights)
    if not args.skip_stage1:
        stage1_best = train_stage(
            ROOT / "dataset_all", "stage1_all", args.stage1_epochs, starter
        )
        candidates.append(("stage1_all", stage1_best))
        starter = stage1_best

    stage2_best = train_stage(
        ROOT / "dataset_photos", "stage2_photos", args.stage2_epochs, starter
    )
    candidates.append(("stage2_photos", stage2_best))

    scores = {}
    for name, path in candidates:
        scores[name] = photo_val_score(path)
        print(f"[{name}] photo-val {scores[name]}")

    winner = max(candidates, key=lambda item: scores[item[0]]["map50_95"])
    final = ROOT / "models" / "thumbs_up_best.pt"
    shutil.copy2(winner[1], final)
    (ROOT / "models" / "model_selection.json").write_text(
        json.dumps(
            {
                "winner": winner[0],
                "scores": scores,
                "weights": str(winner[1]),
            },
            indent=2,
        )
    )
    print(f"winner: {winner[0]} -> {final}")


if __name__ == "__main__":
    main()
