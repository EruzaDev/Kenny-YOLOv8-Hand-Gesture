#!/usr/bin/env python3
"""Generate train/validation curves from the saved Ultralytics CSV logs."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "runs"
OUT = ROOT / "graphs"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
GRAY = "#64748b"
BLUE = "#2563eb"


def _rows(run: str) -> list[dict]:
    path = RUNS / run / "results.csv"
    return list(csv.DictReader(path.open()))


def _values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def _best(rows: list[dict], key: str) -> dict:
    return max(rows, key=lambda row: float(row[key]))


def _save(fig, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def _mark_best(ax, epoch: int) -> None:
    ax.axvline(epoch, color=GREEN, linestyle=":", linewidth=1.2, alpha=0.8)


def plot_classifier(run: str, filename: str) -> dict:
    rows = _rows(run)
    epochs = _values(rows, "epoch").astype(int)
    train_loss = _values(rows, "train/loss")
    val_loss = _values(rows, "val/loss")
    top1 = _values(rows, "metrics/accuracy_top1") * 100
    top5 = _values(rows, "metrics/accuracy_top5") * 100
    best = _best(rows, "metrics/accuracy_top1")
    best_epoch = int(float(best["epoch"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, train_loss, color=BLUE, marker="o", markersize=3, label="Train loss")
    axes[0].plot(epochs, val_loss, color=AMBER, marker="o", markersize=3, label="Validation loss")
    axes[0].set_title("Loss", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(epochs, top1, color=GREEN, marker="o", markersize=3, label="Top-1")
    axes[1].plot(epochs, top5, color=GRAY, marker="o", markersize=3, label="Top-5")
    axes[1].set_title("Validation accuracy", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, fontsize=8)

    gap = val_loss - train_loss
    axes[2].plot(epochs, gap, color=RED, marker="o", markersize=3)
    axes[2].axhline(0, color=GRAY, linewidth=0.8)
    axes[2].set_title("Validation − train loss", weight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss gap")

    for ax in axes:
        _mark_best(ax, best_epoch)
    axes[1].annotate(
        f"best epoch {best_epoch}\ntop-1 {float(best['metrics/accuracy_top1']) * 100:.1f}%",
        xy=(best_epoch, float(best["metrics/accuracy_top1"]) * 100),
        xytext=(best_epoch + 2, max(10, float(best["metrics/accuracy_top1"]) * 100 - 30)),
        fontsize=8,
        color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1},
    )
    fig.suptitle(
        f"YOLOv8n-cls · {run} · {len(rows)} epochs (best checkpoint epoch {best_epoch})",
        weight="bold",
    )
    _save(fig, filename)

    last = rows[-1]
    return {
        "run": run,
        "epochs": len(rows),
        "best_epoch": best_epoch,
        "best_val_top1": float(best["metrics/accuracy_top1"]),
        "best_val_top5": float(best["metrics/accuracy_top5"]),
        "train_loss_at_best": float(best["train/loss"]),
        "val_loss_at_best": float(best["val/loss"]),
        "final_epoch": int(float(last["epoch"])),
        "final_train_loss": float(last["train/loss"]),
        "final_val_loss": float(last["val/loss"]),
        "final_val_top1": float(last["metrics/accuracy_top1"]),
        "final_val_top5": float(last["metrics/accuracy_top5"]),
    }


def plot_detector(run: str, filename: str) -> dict:
    rows = _rows(run)
    epochs = _values(rows, "epoch").astype(int)
    best = _best(rows, "metrics/mAP50-95(B)")
    best_epoch = int(float(best["epoch"]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    losses = (
        ("box", "Box loss"),
        ("cls", "Classification loss"),
        ("dfl", "DFL loss"),
    )
    for ax, (key, title) in zip(axes[0], losses):
        ax.plot(epochs, _values(rows, f"train/{key}_loss"), color=BLUE, marker="o", markersize=2.5, label="Train")
        ax.plot(epochs, _values(rows, f"val/{key}_loss"), color=AMBER, marker="o", markersize=2.5, label="Validation")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=8)
        _mark_best(ax, best_epoch)

    ax = axes[1, 0]
    ax.plot(epochs, _values(rows, "metrics/precision(B)"), color=BLUE, marker="o", markersize=2.5, label="Precision")
    ax.plot(epochs, _values(rows, "metrics/recall(B)"), color=AMBER, marker="o", markersize=2.5, label="Recall")
    ax.set_title("Validation precision / recall", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    _mark_best(ax, best_epoch)

    ax = axes[1, 1]
    ax.plot(epochs, _values(rows, "metrics/mAP50(B)"), color=GREEN, marker="o", markersize=2.5, label="mAP50")
    ax.plot(epochs, _values(rows, "metrics/mAP50-95(B)"), color=RED, marker="o", markersize=2.5, label="mAP50-95")
    ax.set_title("Validation mAP", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    _mark_best(ax, best_epoch)

    ax = axes[1, 2]
    ax.plot(epochs, _values(rows, "lr/pg0"), color=GRAY, marker="o", markersize=2.5)
    ax.set_title("Learning rate", weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    _mark_best(ax, best_epoch)

    fig.suptitle(
        f"YOLOv8n detector · {run} · {len(rows)} epochs (best epoch {best_epoch})",
        weight="bold",
    )
    _save(fig, filename)

    last = rows[-1]
    return {
        "run": run,
        "epochs": len(rows),
        "best_epoch": best_epoch,
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map50_95": float(best["metrics/mAP50-95(B)"]),
        "best_precision": float(best["metrics/precision(B)"]),
        "best_recall": float(best["metrics/recall(B)"]),
        "train_box_loss_at_best": float(best["train/box_loss"]),
        "val_box_loss_at_best": float(best["val/box_loss"]),
        "final_epoch": int(float(last["epoch"])),
        "final_map50": float(last["metrics/mAP50(B)"]),
        "final_map50_95": float(last["metrics/mAP50-95(B)"]),
        "final_train_box_loss": float(last["train/box_loss"]),
        "final_val_box_loss": float(last["val/box_loss"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "classifier": plot_classifier("thumbsup_cls_clean2", "classifier_train_val_curves.png"),
        "crop_detector": plot_detector("thumbsup_crops_n512", "crop_detector_train_val_curves.png"),
        "stage1_detector": plot_detector("stage1_all", "stage1_detector_train_val_curves.png"),
    }
    (OUT / "train_val_summary.json").write_text(json.dumps(summary, indent=2))

    copies = (
        ("thumbsup_cls_clean2", "results.png", "classifier_ultralytics_results.png"),
        ("thumbsup_cls_clean2", "confusion_matrix_normalized.png", "classifier_confusion_matrix_normalized.png"),
        ("thumbsup_crops_n512", "results.png", "crop_detector_ultralytics_results.png"),
        ("thumbsup_crops_n512", "confusion_matrix_normalized.png", "crop_detector_confusion_matrix_normalized.png"),
        ("stage1_all", "results.png", "stage1_detector_ultralytics_results.png"),
        ("stage1_all", "confusion_matrix_normalized.png", "stage1_detector_confusion_matrix_normalized.png"),
    )
    for run, source, target in copies:
        path = RUNS / run / source
        if path.exists():
            shutil.copy2(path, OUT / target)

    print(json.dumps(summary, indent=2))
    print(f"saved training graphs to {OUT}")


if __name__ == "__main__":
    main()
