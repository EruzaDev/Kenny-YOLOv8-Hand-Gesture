#!/usr/bin/env python3
"""Generate result graphs for the thumbs-up pipeline.

Writes confidence distributions, scoring-method comparisons, threshold curves,
confusion matrices, and negative-image checks to ``graphs/``.
"""
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import app as pipeline  # noqa: E402

OUT = ROOT / "graphs"
PHOTO_IMAGES = ROOT / "dataset_teacher_photo_val" / "images"
PHOTO_LABELS = ROOT / "dataset_teacher_photo_val" / "labels"
WEBCAM_IMAGES = ROOT / "dataset_webcam_bench" / "images"
WEBCAM_LABELS = ROOT / "dataset_webcam_bench" / "labels"
TRAIN_IMAGES = ROOT / "dataset_teacher" / "images" / "train"
TRAIN_LABELS = ROOT / "dataset_teacher" / "labels" / "train"

GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
GRAY = "#64748b"
BLUE = "#2563eb"
THRESHOLD = pipeline.DEFAULT_CONFIDENCE


def _nms(detections: list[dict], threshold: float = 0.65) -> list[dict]:
    kept = []
    for det in sorted(detections, key=lambda item: -item["score"]):
        if all(
            pipeline._iou(det["box"], other["box"]) < threshold for other in kept
        ):
            kept.append(det)
    return kept


def _ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
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


def _collect_candidates(image: np.ndarray) -> list[dict]:
    classifier, teacher, _ = pipeline.get_models()
    result = teacher.predict(
        image,
        imgsz=640,
        conf=0.08,
        iou=0.6,
        max_det=12,
        device="cpu",
        verbose=False,
    )[0]
    height, width = image.shape[:2]
    candidates = []
    for box in result.boxes:
        coords = [float(value) for value in box.xyxy[0].tolist()]
        teacher_conf = float(box.conf.item())
        name = teacher.names[int(box.cls.item())]
        classifier_conf = 0.0
        for margin in pipeline.CROP_MARGINS:
            crop = pipeline._crop_hand(image, coords, margin)
            if crop is None:
                continue
            prob = classifier.predict(
                crop, imgsz=224, device="cpu", verbose=False
            )[0].probs
            classifier_conf = max(
                classifier_conf, float(prob.data[pipeline._thumbs_index])
            )
        area = (coords[2] - coords[0]) * (coords[3] - coords[1])
        candidates.append(
            {
                "box": coords,
                "name": name,
                "teacher": teacher_conf,
                "classifier": classifier_conf,
                "area": area / (width * height),
            }
        )
    return candidates


def _load_items(
    image_dir: Path,
    label_dir: Path,
    *,
    positives: bool = True,
    negatives: bool = True,
    photos_only: bool = False,
) -> list[dict]:
    items = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        if photos_only and ".rf." not in stem:
            continue
        image_path = image_dir / f"{stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        truth = _ground_truth(label_path, image.shape[1], image.shape[0])
        if truth and not positives:
            continue
        if not truth and not negatives:
            continue
        items.append(
            {
                "stem": stem,
                "image_path": image_path,
                "truth": truth,
                "candidates": _collect_candidates(image),
            }
        )
    return items


def _final_score(candidate: dict) -> float:
    if candidate["name"] == "like":
        if candidate["teacher"] < pipeline.LIKE_MIN_CONF:
            return 0.0
        return 1 - (1 - candidate["classifier"]) * (1 - candidate["teacher"])
    if (
        candidate["teacher"] <= pipeline.FALLBACK_MAX_HAND_CONF
        and candidate["classifier"] >= 0.5
        and candidate["area"] >= pipeline.FALLBACK_MIN_AREA_FRACTION
    ):
        return candidate["classifier"]
    return 0.0


def _method_detections(item: dict, method: str, threshold: float) -> list[dict]:
    detections = []
    for candidate in item["candidates"]:
        if method == "teacher":
            if candidate["name"] != "like":
                continue
            score = candidate["teacher"]
        elif method == "classifier":
            if candidate["area"] < pipeline.FALLBACK_MIN_AREA_FRACTION:
                continue
            score = candidate["classifier"]
        elif method == "ensemble":
            score = _final_score(candidate)
            if score == 0.0:
                continue
        else:
            raise ValueError(f"unknown method: {method}")
        if score >= threshold:
            detections.append({"box": candidate["box"], "score": score})
    return _nms(detections)


def _match(
    detections: list[dict], truth: list[list[float]]
) -> tuple[list[float], list[float], int]:
    matched = set()
    true_positives = []
    false_positives = []
    for detection in detections:
        best_iou, best_index = 0.0, -1
        for index, box in enumerate(truth):
            value = pipeline._iou(detection["box"], box)
            if value > best_iou:
                best_iou, best_index = value, index
        if best_iou >= 0.5 and best_index not in matched:
            true_positives.append(detection["score"])
            matched.add(best_index)
        else:
            false_positives.append(detection["score"])
    return true_positives, false_positives, len(truth) - len(matched)


def _evaluate(items: list[dict], method: str, threshold: float) -> dict:
    true_positive = []
    false_positive = []
    false_negative = 0
    image_tp = image_fp = image_fn = image_tn = 0
    predicted_images = 0
    per_image = []
    for item in items:
        detections = _method_detections(item, method, threshold)
        tp, fp, fn = _match(detections, item["truth"])
        true_positive.extend(tp)
        false_positive.extend(fp)
        false_negative += fn
        predicted = bool(detections)
        has_truth = bool(item["truth"])
        predicted_images += int(predicted)
        if has_truth and predicted:
            image_tp += 1
        elif has_truth and not predicted:
            image_fn += 1
        elif not has_truth and predicted:
            image_fp += 1
        else:
            image_tn += 1
        if has_truth:
            per_image.append(
                (
                    item["stem"],
                    max((det["score"] for det in detections), default=0.0),
                )
            )
    tp_count = len(true_positive)
    fp_count = len(false_positive)
    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
    recall = tp_count / (tp_count + false_negative) if tp_count + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp_count,
        "fp": fp_count,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_scores": true_positive,
        "fp_scores": false_positive,
        "predicted_images": predicted_images,
        "images": len(items),
        "image_tp": image_tp,
        "image_fp": image_fp,
        "image_fn": image_fn,
        "image_tn": image_tn,
        "per_image": per_image,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _save(fig, name: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def plot_confidence(photo_tp: list[float], webcam_tp: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, scores, title in (
        (axes[0], photo_tp, "Photo validation"),
        (axes[1], webcam_tp, "Webcam-like crops"),
    ):
        ax.hist(scores, bins=np.arange(0.0, 1.01, 0.05), color=GREEN, edgecolor="white")
        ax.axvline(THRESHOLD, color=RED, linestyle="--", linewidth=1.4)
        ax.set_title(f"{title} (n={len(scores)})", fontsize=10, weight="bold")
        ax.set_xlabel("Final thumbs-up confidence")
        ax.set_xlim(0.0, 1.0)
        ax.text(
            0.02,
            0.96,
            f"≥ {THRESHOLD:.0%}: {sum(score >= THRESHOLD for score in scores)}/{len(scores)}",
            transform=ax.transAxes,
            va="top",
            color=GREEN,
            fontsize=9,
        )
    axes[0].set_ylabel("Thumbs-up instances")
    fig.suptitle("Confidence score distribution (true positives)", weight="bold")
    _save(fig, "confidence_distribution.png")


def plot_method_comparison(results: dict) -> None:
    methods = ("teacher", "classifier", "ensemble")
    labels = ("Teacher", "Classifier", "Ensemble")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, dataset, title in (
        (axes[0], "photo_val", "Photo validation"),
        (axes[1], "webcam", "Webcam-like crops"),
    ):
        x = np.arange(len(methods))
        tp = np.array([results[dataset][method]["tp"] for method in methods])
        fn = np.array([results[dataset][method]["fn"] for method in methods])
        fp = np.array([results[dataset][method]["fp"] for method in methods])
        ax.bar(x, tp, color=GREEN, label="Correct detections")
        ax.bar(x, fn, bottom=tp, color=AMBER, label="Missed thumbs up")
        ax.bar(x, fp, bottom=tp + fn, color=RED, label="False detections")
        for index, method in enumerate(methods):
            score = results[dataset][method]["f1"]
            ax.text(index, tp[index] + fn[index] + fp[index] + 0.35,
                    f"F1 {score:.2f}", ha="center", fontsize=9, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 31)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Thumbs-up instances / detections")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"What each scoring method found · confidence ≥ {THRESHOLD:.0%} · IoU ≥ 0.50", weight="bold")
    _save(fig, "score_method_comparison.png")


def plot_threshold_curve(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset, title in (
        (axes[0], "photo_val", "Photo validation"),
        (axes[1], "webcam", "Webcam-like crops"),
    ):
        thresholds = sorted(float(value) for value in sweep[dataset])
        for metric, color in (("precision", BLUE), ("recall", AMBER), ("f1", GREEN)):
            values = [sweep[dataset][f"{threshold:.2f}"][metric] for threshold in thresholds]
            ax.plot(thresholds, values, marker="o", markersize=3, label=metric.upper(), color=color)
        ax.axvline(THRESHOLD, color=RED, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Score")
    fig.suptitle("Ensemble metrics vs confidence threshold", weight="bold")
    _save(fig, "metrics_vs_threshold.png")


def plot_confusion(photo_val: dict, combined: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], photo_val, "Photo validation (19 pos / 11 neg)"),
        (axes[1], combined, "Photo val + 50 train negatives"),
    ):
        grid = np.array(
            [
                [matrix["image_tp"], matrix["image_fn"]],
                [matrix["image_fp"], matrix["image_tn"]],
            ]
        )
        image = ax.imshow(grid, cmap="Greens", vmin=0)
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    int(grid[row, col]),
                    ha="center",
                    va="center",
                    fontsize=18,
                    weight="bold",
                    color="white" if grid[row, col] > grid.max() / 2 else "#0f172a",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Detected", "Not detected"])
        ax.set_yticklabels(["Has thumbs up", "No thumbs up"])
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        fig.colorbar(image, ax=ax, shrink=0.7, label="Images")
    fig.suptitle(f"Image-level confusion matrix at {THRESHOLD:.0%} confidence", weight="bold")
    _save(fig, "confusion_matrix.png", tight=False)


def plot_scatter(items: list[dict]) -> None:
    points = []
    for item in items:
        for candidate in item["candidates"]:
            score = _final_score(candidate)
            best_iou = max(
                (pipeline._iou(candidate["box"], box) for box in item["truth"]),
                default=0.0,
            )
            points.append(
                {
                    "teacher": candidate["teacher"],
                    "classifier": candidate["classifier"],
                    "accepted": score >= THRESHOLD,
                    "matched": best_iou >= 0.5,
                }
            )
    categories = (
        ("Accepted TP", GREEN, lambda point: point["accepted"] and point["matched"]),
        ("Accepted FP", RED, lambda point: point["accepted"] and not point["matched"]),
        ("Rejected", GRAY, lambda point: not point["accepted"]),
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, color, predicate in categories:
        xs = [point["teacher"] for point in points if predicate(point)]
        ys = [point["classifier"] for point in points if predicate(point)]
        ax.scatter(xs, ys, s=26, alpha=0.75, color=color, edgecolor="white", linewidth=0.4, label=label)
    xs = np.linspace(pipeline.LIKE_MIN_CONF, THRESHOLD - 0.001, 100)
    ys = 1 - (1 - THRESHOLD) / (1 - xs)
    ax.plot(xs, ys, linestyle="--", color=GREEN, linewidth=1.2, label=f"Noisy-OR = {THRESHOLD:.0%}")
    ax.axvline(pipeline.LIKE_MIN_CONF, color=BLUE, linestyle=":", linewidth=1.2)
    ax.axhline(0.5, color=AMBER, linestyle=":", linewidth=1.2)
    ax.set_xlabel("Teacher 'like' confidence")
    ax.set_ylabel("Classifier thumbs-up probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Photo validation: teacher vs classifier scores", weight="bold")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _save(fig, "teacher_vs_classifier.png")


def plot_per_image(photo: dict, webcam: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dataset, title in (
        (axes[0], photo, "Photo validation"),
        (axes[1], webcam, "Webcam-like crops"),
    ):
        rows = sorted(dataset["per_image"], key=lambda item: -item[1])
        labels = [stem.split("_")[0] for stem, _ in rows]
        values = [score for _, score in rows]
        colors = [GREEN if score >= THRESHOLD else AMBER for score in values]
        bars = ax.bar(range(len(values)), values, color=colors)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
        ax.axhline(THRESHOLD, color=RED, linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_ylabel("Top confidence")
    fig.suptitle("Top confidence per positive image", weight="bold")
    _save(fig, "per_image_confidence.png")


def plot_negative_check(total: int, at_30: int, at_default: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = [at_30, at_default]
    bars = ax.barh(["30% threshold", f"{THRESHOLD:.0%} threshold"], values,
                   color=[AMBER, GREEN], height=0.5)
    for bar, value in zip(bars, values):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{value}/{total} images ({value / total:.1%})", va="center", fontsize=10)
    ax.set_xlim(0, max(2, max(values) + 0.6))
    ax.set_xlabel("Negative images with a false thumbs-up detection")
    ax.set_title(f"False alarms on {total} negative photos", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    _save(fig, "negative_photo_check.png")


def _confidence_stats(scores: list[float]) -> dict:
    values = np.array(scores, dtype=float)
    return {
        "n": int(len(values)),
        "at_least_default": int((values >= THRESHOLD).sum()),
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "min": float(values.min()) if len(values) else 0.0,
        "max": float(values.max()) if len(values) else 0.0,
    }


def plot_optimization(sweep: dict, negative_items: list[dict]) -> dict:
    """Show the measured threshold tradeoff and export its underlying counts."""
    rows = []
    for threshold_text in sorted(sweep["photo_val"], key=float):
        threshold = float(threshold_text)
        negatives = _evaluate(negative_items, "ensemble", threshold)
        row = {"threshold": threshold,
               "negative_images": negatives["images"],
               "negative_false_alarm_images": negatives["predicted_images"]}
        for dataset in ("photo_val", "webcam"):
            result = sweep[dataset][threshold_text]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                row[f"{dataset}_{key}"] = result[key]
        rows.append(row)
    with (OUT / "threshold_optimization.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    thresholds = [row["threshold"] for row in rows]
    for dataset, color, label in (("photo_val", BLUE, "Photo validation"),
                                  ("webcam", GREEN, "Webcam benchmark")):
        axes[0].plot(thresholds, [row[f"{dataset}_recall"] for row in rows],
                     marker="o", markersize=3, color=color, label=label)
    axes[0].set_ylabel("Thumbs-up recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    axes[1].step(thresholds, [row["negative_false_alarm_images"] for row in rows],
                 where="post", color=RED, linewidth=2)
    axes[1].set_ylabel(f"False-alarm images / {len(negative_items)} negatives")
    axes[1].set_ylim(-0.1, max(row["negative_false_alarm_images"] for row in rows) + 0.5)
    for ax in axes:
        ax.axvline(THRESHOLD, color=AMBER, linestyle="--", label=f"Default {THRESHOLD:.0%}")
        ax.set_xlabel("Minimum prediction confidence")
        ax.grid(alpha=0.2)
    fig.suptitle("Threshold choice: thumbs-up recall and negative-photo false alarms", weight="bold")
    _save(fig, "threshold_optimization.png")
    return {f"{row['threshold']:.2f}": row for row in rows}


def export_image_diagnostics(groups: dict[str, list[dict]]) -> dict:
    """Save auditable per-image results and a gallery of hard examples."""
    rows = []
    examples = []
    visualizer_records = []
    for dataset, items in groups.items():
        for item in items:
            detections = _method_detections(item, "ensemble", THRESHOLD)
            tp, fp, fn = _match(detections, item["truth"])
            row = {
                "dataset": dataset, "image": item["image_path"].relative_to(ROOT).as_posix(),
                "ground_truth_boxes": len(item["truth"]), "predicted_boxes": len(detections),
                "true_positives": len(tp), "false_positives": len(fp), "false_negatives": fn,
                "top_confidence": max((det["score"] for det in detections), default=0.0),
                "candidate_boxes": len(item["candidates"]),
            }
            rows.append(row)
            image = cv2.imread(str(item["image_path"]))
            visualizer_records.append({
                **row, "width": int(image.shape[1]), "height": int(image.shape[0]),
                "truth": item["truth"], "detections": detections,
            })
            if fn or fp:
                examples.append((0, dataset, item, detections, row))
            elif item["truth"] and detections:
                examples.append((1, dataset, item, detections, row))
    with (OUT / "per_image_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_visualizer(visualizer_records)

    # Failures first, followed by the lowest-confidence correct detections.
    examples.sort(key=lambda entry: (entry[0], entry[4]["top_confidence"]))
    selected = examples[:12]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, dataset, item, detections, row) in zip(axes.flat, selected):
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            continue
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for box in item["truth"]:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=GREEN, linewidth=2))
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=BLUE, linewidth=2, linestyle="--"))
            ax.text(x1, max(0, y1 - 2), f"{det['score']:.2f}", color="white",
                    fontsize=8, bbox={"facecolor": BLUE, "edgecolor": "none", "pad": 1})
        ax.set_title(f"{dataset}: {item['stem'][:18]}\nTP {row['true_positives']}  FP {row['false_positives']}  FN {row['false_negatives']}",
                     fontsize=9, color=RED if row["false_negatives"] or row["false_positives"] else "black")
        ax.axis("off")
    fig.suptitle(f"Detection examples at {THRESHOLD:.0%} · green = ground truth · dashed blue = prediction", weight="bold")
    _save(fig, "detection_examples.png")
    return {"images": len(rows), "shown_examples": len(selected),
            "failure_images": sum(bool(row["false_positives"] or row["false_negatives"]) for row in rows)}


def _write_visualizer(records: list[dict]) -> None:
    """Build a portable local image viewer with prediction/label overlays."""
    payload = json.dumps(records).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thumbs-up evaluation viewer</title>
<style>
body{font:15px system-ui,sans-serif;background:#0b1220;color:#eef4ff;margin:0;padding:24px;max-width:1100px;margin:auto}
h1{margin:0 0 8px}p{color:#a8b8cd}a{color:#93c5fd}select,button,input{background:#1d2b40;color:white;border:1px solid #52647e;border-radius:7px;padding:9px}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:20px 0}.bar input{min-width:260px}
.stage{background:#151f30;padding:14px;border-radius:12px}svg{display:block;width:100%;max-height:70vh;background:#101827}
.details{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0;color:#bed0e6}.ok{color:#4ade80}.bad{color:#fb7185}
.legend span{display:inline-block;margin-right:18px}.swatch{display:inline-block;width:15px;height:15px;margin-right:5px;vertical-align:-2px;border:3px solid}
</style>
<h1>Thumbs-up evaluation viewer</h1>
<p>Inspect every evaluated image. Green solid boxes are labels; blue dashed boxes are predictions. Prediction numbers are model confidence, not accuracy.</p>
<div class="bar"><select id="dataset"><option value="all">All datasets</option><option value="photo_val">Photo validation</option><option value="webcam">Webcam benchmark</option><option value="train_negatives">Training negatives</option></select>
<select id="outcome"><option value="all">All outcomes</option><option value="errors">Errors only</option><option value="correct">Correct detections</option><option value="negative">No thumbs up</option></select>
<input id="search" placeholder="Search filename"><button id="prev">Previous</button><button id="next">Next</button><span id="position"></span></div>
<div class="stage"><svg id="scene" xmlns="http://www.w3.org/2000/svg"></svg><div class="details" id="details"></div>
<div class="legend"><span><i class="swatch" style="border-color:#4ade80"></i>Ground truth</span><span><i class="swatch" style="border-color:#60a5fa"></i>Prediction</span></div></div>
<p><a href="per_image_results.csv">Download per-image results</a> · <a href="THUMBS_UP_EVALUATION.md">Read evaluation report</a></p>
<script>const records=__DATA__;let shown=records,position=0;
const $=id=>document.getElementById(id);const ns='http://www.w3.org/2000/svg';
function filter(){const ds=$('dataset').value,oc=$('outcome').value,q=$('search').value.toLowerCase();shown=records.filter(r=>(ds==='all'||r.dataset===ds)&&r.image.toLowerCase().includes(q)&&(oc==='all'||(oc==='errors'&&(r.false_positives||r.false_negatives))||(oc==='correct'&&r.true_positives)||(oc==='negative'&&!r.ground_truth_boxes)));position=0;render()}
function element(tag,attrs){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}
function render(){const svg=$('scene');svg.replaceChildren();if(!shown.length){$('details').textContent='No images match the filters.';$('position').textContent='0 / 0';return}const r=shown[position];svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);svg.append(element('image',{href:'../'+r.image,x:0,y:0,width:r.width,height:r.height}));
for(const box of r.truth){const [x1,y1,x2,y2]=box;svg.append(element('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:'#4ade80','stroke-width':Math.max(2,r.width/250)}))}
for(const det of r.detections){const [x1,y1,x2,y2]=det.box;svg.append(element('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:'#60a5fa','stroke-width':Math.max(2,r.width/250),'stroke-dasharray':'8 5'}));const t=element('text',{x:x1+4,y:Math.max(18,y1-5),fill:'#fff','font-size':Math.max(16,r.width/35),'font-weight':'bold'});t.textContent=(det.score*100).toFixed(1)+'%';svg.append(t)}
$('position').textContent=`${position+1} / ${shown.length}`;$('details').replaceChildren();for(const text of [r.dataset,r.image.split('/').pop(),`TP ${r.true_positives}`,`FP ${r.false_positives}`,`FN ${r.false_negatives}`,`Candidates ${r.candidate_boxes}`]){const s=document.createElement('span');s.textContent=text;if(text.startsWith('FP ')&&r.false_positives||text.startsWith('FN ')&&r.false_negatives)s.className='bad';$('details').append(s)}}
for(const id of ['dataset','outcome','search'])$(id).addEventListener(id==='search'?'input':'change',filter);$('prev').onclick=()=>{if(shown.length){position=(position+shown.length-1)%shown.length;render()}};$('next').onclick=()=>{if(shown.length){position=(position+1)%shown.length;render()}};render();</script></html>'''
    (OUT / "evaluation_viewer.html").write_text(html.replace("__DATA__", payload))


def plot_outcomes(groups: dict[str, list[dict]]) -> None:
    """Compare box-level outcomes without hiding the webcam misses."""
    names = ("Photo validation", "Webcam benchmark")
    results = [_evaluate(groups[key], "ensemble", THRESHOLD) for key in ("photo_val", "webcam")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(2)
    tp = np.array([result["tp"] for result in results])
    fn = np.array([result["fn"] for result in results])
    fp = np.array([result["fp"] for result in results])
    ax.bar(x, tp, color=GREEN, label="Correct")
    ax.bar(x, fn, bottom=tp, color=AMBER, label="Missed")
    ax.bar(x, fp, bottom=tp + fn, color=RED, label="False detection")
    for index, result in enumerate(results):
        ax.text(index, tp[index] + fn[index] + fp[index] + 0.3,
                f"Recall {result['recall']:.1%}  ·  F1 {result['f1']:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(tp + fn + fp) + 4)
    ax.set_ylabel("Thumbs-up boxes")
    ax.legend(frameon=False)
    ax.set_title(f"Detection outcomes at {THRESHOLD:.0%} confidence", weight="bold")
    _save(fig, "detection_outcomes.png")


def plot_thumbs_up_performance(photo: dict, webcam: dict) -> None:
    """Present only the final app's thumbs-up detection metrics."""
    fig, ax = plt.subplots(figsize=(9, 5))
    datasets = ("Photo validation", "Webcam crops")
    x = np.arange(2)
    for offset, key, label, color in ((-0.23, "precision", "Precision", BLUE),
                                       (0, "recall", "Recall", AMBER),
                                       (0.23, "f1", "F1", GREEN)):
        bars = ax.bar(x + offset, [photo[key], webcam[key]], 0.22, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in (photo[key], webcam[key])], padding=3)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Thumbs-up detection score")
    ax.set_title(f"Final app · thumbs-up performance at {THRESHOLD:.0%} confidence", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    _save(fig, "thumbs_up_performance.png")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    z = 1.96
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(center - half), float(center + half)


def plot_uncertainty(photo: dict, webcam: dict, negative_val: dict, negative_train: dict) -> dict:
    samples = [
        ("Photo box recall", photo["tp"], photo["tp"] + photo["fn"]),
        ("Webcam box recall", webcam["tp"], webcam["tp"] + webcam["fn"]),
        ("Photo negative specificity", negative_val["image_tn"], negative_val["images"]),
        ("Train negative specificity", negative_train["image_tn"], negative_train["images"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    output = {}
    for index, (name, successes, total) in enumerate(samples):
        low, high = _wilson(successes, total)
        value = successes / total
        ax.plot([low, high], [index, index], color=BLUE, linewidth=5, solid_capstyle="round")
        ax.scatter([value], [index], color=GREEN, s=70, zorder=3)
        ax.text(min(1.02, value + 0.015), index - 0.13, f"{successes}/{total}", fontsize=9)
        output[name] = {"successes": successes, "total": total, "rate": value,
                        "wilson_95_low": low, "wilson_95_high": high}
    ax.set_yticks(range(len(samples)), [name for name, _, _ in samples])
    ax.set_xlim(0.55, 1.13)
    ax.set_ylim(len(samples) - 0.5, -0.5)
    ax.set_xlabel("Observed rate (dot) and Wilson 95% interval (line)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Sample-size uncertainty · training negatives are not independent", weight="bold")
    _save(fig, "evaluation_uncertainty.png")
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline.get_models()

    photo_items = _load_items(PHOTO_IMAGES, PHOTO_LABELS)
    webcam_items = _load_items(WEBCAM_IMAGES, WEBCAM_LABELS)
    negative_val = [item for item in photo_items if not item["truth"]]
    negative_train = _load_items(
        TRAIN_IMAGES, TRAIN_LABELS, positives=False, photos_only=True
    )
    negative_items = negative_val + negative_train
    diagnostics = export_image_diagnostics({"photo_val": photo_items, "webcam": webcam_items,
                                            "train_negatives": negative_train})
    plot_outcomes({"photo_val": photo_items, "webcam": webcam_items})

    methods = ("teacher", "classifier", "ensemble")
    method_results = {
        "photo_val": {
            method: _evaluate(photo_items, method, THRESHOLD) for method in methods
        },
        "webcam": {
            method: _evaluate(webcam_items, method, THRESHOLD) for method in methods
        },
    }

    thresholds = [round(value, 2) for value in np.arange(0.30, 0.96, 0.05)]
    sweep = {
        "photo_val": {
            f"{threshold:.2f}": _evaluate(photo_items, "ensemble", threshold)
            for threshold in thresholds
        },
        "webcam": {
            f"{threshold:.2f}": _evaluate(webcam_items, "ensemble", threshold)
            for threshold in thresholds
        },
    }
    optimization = plot_optimization(sweep, negative_items)

    photo_at_zero = _evaluate(photo_items, "ensemble", 0.0)
    webcam_at_zero = _evaluate(webcam_items, "ensemble", 0.0)
    photo_at_default = _evaluate(photo_items, "ensemble", THRESHOLD)
    webcam_at_default = _evaluate(webcam_items, "ensemble", THRESHOLD)
    negative_at_30 = _evaluate(negative_items, "ensemble", 0.30)
    negative_at_default = _evaluate(negative_items, "ensemble", THRESHOLD)
    negative_val_at_default = _evaluate(negative_val, "ensemble", THRESHOLD)
    negative_train_at_default = _evaluate(negative_train, "ensemble", THRESHOLD)
    combined_items = photo_items + negative_train
    combined_at_default = _evaluate(combined_items, "ensemble", THRESHOLD)
    plot_thumbs_up_performance(photo_at_default, webcam_at_default)
    uncertainty = plot_uncertainty(photo_at_default, webcam_at_default,
                                   negative_val_at_default, negative_train_at_default)

    plot_confidence(photo_at_zero["tp_scores"], webcam_at_zero["tp_scores"])
    plot_method_comparison(method_results)
    plot_threshold_curve(sweep)
    plot_confusion(photo_at_default, combined_at_default)
    plot_scatter(photo_items)
    plot_per_image(photo_at_zero, webcam_at_zero)
    plot_negative_check(
        len(negative_items),
        negative_at_30["predicted_images"],
        negative_at_default["predicted_images"],
    )

    summary = {
        "datasets": {
            "photo_val": {
                "images": len(photo_items),
                "positive_images": sum(bool(item["truth"]) for item in photo_items),
                "negative_images": sum(not item["truth"] for item in photo_items),
                "ground_truth_thumbs_up": sum(
                    len(item["truth"]) for item in photo_items
                ),
            },
            "webcam": {
                "images": len(webcam_items),
                "positive_images": sum(bool(item["truth"]) for item in webcam_items),
                "ground_truth_thumbs_up": sum(
                    len(item["truth"]) for item in webcam_items
                ),
            },
            "negatives": {"images": len(negative_items)},
        },
        "threshold": THRESHOLD,
        "score_methods": {
            dataset: {
                method: {
                    key: result[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                }
                for method, result in per_method.items()
            }
            for dataset, per_method in method_results.items()
        },
        "threshold_sweep_ensemble": {
            dataset: {
                threshold: {
                    key: result[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                }
                for threshold, result in per_threshold.items()
            }
            for dataset, per_threshold in sweep.items()
        },
        "confusion_image_level": {
            "photo_val": {
                key: photo_at_default[key]
                for key in ("image_tp", "image_fp", "image_fn", "image_tn")
            },
            "photo_val_plus_train_negatives": {
                key: combined_at_default[key]
                for key in ("image_tp", "image_fp", "image_fn", "image_tn")
            },
        },
        "confidence": {
            "photo_val": _confidence_stats(photo_at_zero["tp_scores"]),
            "webcam": _confidence_stats(webcam_at_zero["tp_scores"]),
        },
        "negative_check": {
            "images": len(negative_items),
            "with_detections_at_0_30": negative_at_30["predicted_images"],
            "with_detections_at_default": negative_at_default["predicted_images"],
            "object_false_positives_at_0_30": negative_at_30["fp"],
            "object_false_positives_at_default": negative_at_default["fp"],
        },
        "threshold_optimization": optimization,
        "diagnostics": diagnostics,
        "uncertainty": uncertainty,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["score_methods"], indent=2))
    print(json.dumps(summary["confidence"], indent=2))
    print(json.dumps(summary["negative_check"], indent=2))
    print(f"saved graphs to {OUT}")


if __name__ == "__main__":
    main()
