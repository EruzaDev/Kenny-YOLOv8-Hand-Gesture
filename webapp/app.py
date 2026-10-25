#!/usr/bin/env python3
"""ThumbScan - real-time thumbs-up detector.

The detector is a two-stage pipeline:
  1. A hand detector finds every hand and its rough gesture.
  2. A YOLOv8 classifier re-checks a close-up crop of each hand.

Running the classifier on the crop (instead of the whole frame) is what makes
the confidence high enough for live use.  The hand detector's non-thumbs-up
label also suppresses classifier false positives.
"""
from __future__ import annotations

import base64
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = Path(
    os.environ.get("YOLO_CLASSIFIER", ROOT / "models" / "thumbs_up_classifier.pt")
)
TEACHER_PATH = Path(
    os.environ.get("YOLO_TEACHER", ROOT / "models" / "teacher_haGRID_yolo26n.pt")
)
DETECTOR_PATH = Path(os.environ.get("YOLO_MODEL", ROOT / "models" / "thumbs_up_best.pt"))
CLASSIFIER_FALLBACKS = [
    ROOT / "models" / "runs" / "thumbsup_cls_clean2" / "weights" / "best.pt",
    ROOT / "models" / "runs" / "thumbsup_cls_n224" / "weights" / "best.pt",
]
DETECTOR_FALLBACKS = [
    ROOT / "models" / "runs" / "stage2_photos" / "weights" / "best.pt",
    ROOT / "models" / "runs" / "stage1_all" / "weights" / "best.pt",
]

LABELS = {"thumbs_up": "Thumbs Up"}
COLORS = {"thumbs_up": "#22c55e"}
LIKE_MIN_CONF = 0.35
FALLBACK_MAX_HAND_CONF = 0.25
FALLBACK_MIN_AREA_FRACTION = 0.02
CROP_MARGINS = (0.4, 0.7, 1.0)
DEFAULT_CONFIDENCE = 0.80

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

_lock = threading.Lock()
_classifier: YOLO | None = None
_teacher: YOLO | None = None
_detector: YOLO | None = None
_thumbs_index: int | None = None
_model_error: str | None = None


def _first_existing(path: Path, fallbacks: list[Path]) -> Path | None:
    for candidate in [path, *fallbacks]:
        if candidate.exists():
            return candidate
    return None


def _load(path: Path | None, role: str) -> YOLO:
    if path is None:
        raise RuntimeError(f"No {role} model found. Run training first.")
    model = YOLO(str(path))
    model.to("cpu")
    return model


def get_models() -> tuple[YOLO, YOLO | None, YOLO | None]:
    """Return (classifier, teacher, fallback detector). Model loading is lazy."""
    global _classifier, _teacher, _detector, _thumbs_index, _model_error
    if _classifier is not None:
        return _classifier, _teacher, _detector
    try:
        cls_path = _first_existing(CLASSIFIER_PATH, CLASSIFIER_FALLBACKS)
        _classifier = _load(cls_path, "classifier")
        _thumbs_index = next(
            (i for i, name in _classifier.names.items() if name == "thumbs_up"), None
        )
        if _thumbs_index is None:
            raise RuntimeError("Classifier has no thumbs_up class.")

        teacher_path = _first_existing(TEACHER_PATH, [])
        _teacher = _load(teacher_path, "hand detector") if teacher_path else None
        if _teacher is None:
            _detector = _load(
                _first_existing(DETECTOR_PATH, DETECTOR_FALLBACKS), "detector"
            )
    except RuntimeError as exc:
        _model_error = str(exc)
        raise
    return _classifier, _teacher, _detector


def decode_image() -> np.ndarray | None:
    if "file" in request.files:
        raw = request.files["file"].read()
    else:
        payload = request.get_json(silent=True) or {}
        raw_b64 = payload.get("image", "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        if not raw_b64:
            return None
        try:
            raw = base64.b64decode(raw_b64)
        except Exception:
            return None
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _crop_hand(image: np.ndarray, box: list[float], margin: float = 0.7) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(width, int(cx + side / 2))
    bottom = min(height, int(cy + side / 2))
    if right - left < 16 or bottom - top < 16:
        return None
    return cv2.resize(
        image[top:bottom, left:right], (640, 640), interpolation=cv2.INTER_AREA
    )


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms(detections: list[dict], threshold: float = 0.65) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(detections, key=lambda d: -d["confidence"]):
        if all(_iou(det["box"], other["box"]) < threshold for other in kept):
            kept.append(det)
    return kept


def _make_detection(
    class_id: int,
    raw_label: str,
    confidence: float,
    box: list[float],
    width: int,
    height: int,
) -> dict:
    x1, y1, x2, y2 = box
    return {
        "class_id": class_id,
        "label": LABELS.get(raw_label, raw_label.replace("_", " ").title()),
        "raw_label": raw_label,
        "color": COLORS.get(raw_label, "#f59e0b"),
        "confidence": round(float(confidence), 4),
        "box": [
            round(max(0.0, x1), 2),
            round(max(0.0, y1), 2),
            round(min(float(width), x2), 2),
            round(min(float(height), y2), 2),
        ],
    }


def run_pipeline(image: np.ndarray, conf: float) -> list[dict]:
    classifier, teacher, detector = get_models()
    height, width = image.shape[:2]
    detections: list[dict] = []

    if teacher is not None:
        hands = teacher.predict(
            image,
            imgsz=640,
            conf=0.08,
            iou=0.6,
            max_det=12,
            device="cpu",
            verbose=False,
        )[0]
        for box in hands.boxes:
            coords = [float(v) for v in box.xyxy[0].tolist()]
            name = teacher.names[int(box.cls.item())]
            hand_conf = float(box.conf.item())
            class_conf = 0.0
            for margin in CROP_MARGINS:
                crop = _crop_hand(image, coords, margin)
                if crop is None:
                    continue
                probs = classifier.predict(
                    crop, imgsz=224, device="cpu", verbose=False
                )[0].probs
                class_conf = max(class_conf, float(probs.data[_thumbs_index]))

            if name == "like":
                if hand_conf < LIKE_MIN_CONF:
                    continue
                score = 1 - (1 - class_conf) * (1 - hand_conf)
                if score >= conf:
                    detections.append(
                        _make_detection(0, "thumbs_up", score, coords, width, height)
                    )
            elif (
                hand_conf <= FALLBACK_MAX_HAND_CONF
                and class_conf >= 0.5
                and class_conf >= conf
                and (coords[2] - coords[0]) * (coords[3] - coords[1])
                >= FALLBACK_MIN_AREA_FRACTION * width * height
            ):
                detections.append(
                    _make_detection(0, "thumbs_up", class_conf, coords, width, height)
                )
    elif detector is not None:
        result = detector.predict(
            image, imgsz=512, conf=conf, iou=0.5, device="cpu", verbose=False
        )[0]
        for box in result.boxes:
            name = result.names[int(box.cls.item())]
            if name != "thumbs_up":
                continue
            detections.append(
                _make_detection(
                    0,
                    name,
                    float(box.conf.item()),
                    [float(v) for v in box.xyxy[0].tolist()],
                    width,
                    height,
                )
            )

    detections = _nms(detections)
    detections.sort(key=lambda d: -d["confidence"])
    return detections


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    try:
        classifier, teacher, detector = get_models()
        return jsonify(
            {
                "ok": True,
                "classifier": CLASSIFIER_PATH.name,
                "hand_detector": TEACHER_PATH.name if teacher else "fallback detector",
                "mode": "teacher+classifier" if teacher else "single-stage",
                "error": None,
            }
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)})


@app.post("/predict")
def predict():
    try:
        conf = float(request.form.get("conf", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        conf = DEFAULT_CONFIDENCE
    conf = min(max(conf, 0.05), 0.95)

    image = decode_image()
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        get_models()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    with _lock:
        detections = run_pipeline(image, conf)

    height, width = image.shape[:2]
    top = max((d["confidence"] for d in detections), default=0.0)
    return jsonify(
        {
            "detections": detections,
            "width": width,
            "height": height,
            "count": len(detections),
            "thumbs_up": bool(detections),
            "top_confidence": top,
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Image is too large (max 12 MB)"}), 413


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting ThumbScan on http://localhost:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
