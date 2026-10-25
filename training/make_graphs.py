#!/usr/bin/env python3
"""Generate result graphs for the thumbs-up pipeline.

Writes confidence distributions, scoring-method comparisons, threshold curves,
confusion matrices, and negative-image checks to ``graphs/``.
"""
from __future__ import annotations

import json
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
THRESHOLD = 0.90


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
            f"≥ 90%: {sum(score >= THRESHOLD for score in scores)}/{len(scores)}",
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
    labels = ("Teacher only", "Classifier only", "Noisy-OR ensemble")
    colors = (BLUE, AMBER, GREEN)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, dataset, title in (
        (axes[0], "photo_val", "Photo validation"),
        (axes[1], "webcam", "Webcam-like crops"),
    ):
        x = np.arange(len(methods))
        width = 0.24
        for offset, metric in enumerate(("precision", "recall", "f1")):
            values = [results[dataset][method][metric] for method in methods]
            bars = ax.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=metric.upper(),
                color=colors[offset],
                alpha=0.9,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.0, 1.12)
        ax.set_title(f"{title} · threshold 90%", fontsize=10, weight="bold")
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    axes[0].set_ylabel("Score")
    fig.suptitle("Scoring-method comparison", weight="bold")
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
                [matrix["image_tp"], matrix["image_fp"]],
                [matrix["image_fn"], matrix["image_tn"]],
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
    fig.suptitle("Image-level confusion matrix at 90% confidence", weight="bold")
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
    xs = np.linspace(pipeline.LIKE_MIN_CONF, 0.899, 100)
    ys = 1 - (1 - THRESHOLD) / (1 - xs)
    ax.plot(xs, ys, linestyle="--", color=GREEN, linewidth=1.2, label="Noisy-OR = 90%")
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


def plot_negative_check(total: int, at_30: int, at_90: int) -> None:
    labels = ["No thumbs up", "False @ 30%", "False @ 90%"]
    values = [total, at_30, at_90]
    colors = [GRAY, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.4,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.set_ylabel("Images")
    ax.set_ylim(0, max(values) + 4)
    ax.set_title(f"Negative-only photo check (n={total})", weight="bold")
    _save(fig, "negative_photo_check.png")


def _confidence_stats(scores: list[float]) -> dict:
    values = np.array(scores, dtype=float)
    return {
        "n": int(len(values)),
        "at_least_0_90": int((values >= THRESHOLD).sum()),
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "min": float(values.min()) if len(values) else 0.0,
        "max": float(values.max()) if len(values) else 0.0,
    }


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

    photo_at_zero = _evaluate(photo_items, "ensemble", 0.0)
    webcam_at_zero = _evaluate(webcam_items, "ensemble", 0.0)
    photo_at_90 = _evaluate(photo_items, "ensemble", THRESHOLD)
    webcam_at_90 = _evaluate(webcam_items, "ensemble", THRESHOLD)
    negative_at_30 = _evaluate(negative_items, "ensemble", 0.30)
    negative_at_90 = _evaluate(negative_items, "ensemble", THRESHOLD)
    combined_items = photo_items + negative_train
    combined_at_90 = _evaluate(combined_items, "ensemble", THRESHOLD)

    plot_confidence(photo_at_zero["tp_scores"], webcam_at_zero["tp_scores"])
    plot_method_comparison(method_results)
    plot_threshold_curve(sweep)
    plot_confusion(photo_at_90, combined_at_90)
    plot_scatter(photo_items)
    plot_per_image(photo_at_zero, webcam_at_zero)
    plot_negative_check(
        len(negative_items),
        negative_at_30["predicted_images"],
        negative_at_90["predicted_images"],
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
                key: photo_at_90[key]
                for key in ("image_tp", "image_fp", "image_fn", "image_tn")
            },
            "photo_val_plus_train_negatives": {
                key: combined_at_90[key]
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
            "with_detections_at_0_90": negative_at_90["predicted_images"],
            "object_false_positives_at_0_30": negative_at_30["fp"],
            "object_false_positives_at_0_90": negative_at_90["fp"],
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["score_methods"], indent=2))
    print(json.dumps(summary["confidence"], indent=2))
    print(json.dumps(summary["negative_check"], indent=2))
    print(f"saved graphs to {OUT}")


if __name__ == "__main__":
    main()
