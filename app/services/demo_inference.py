"""Inference helpers and per-step metrics for demo endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.services.behavior_labels import BEHAVIOR_LABELS
from app.services.behavior_model import BehaviorModelFallback, BehaviorModelRuntime
from app.services.object_tracking import (
    ObjectDetector,
    SimpleObjectTracker,
    draw_tracked_objects,
    encode_jpg_data_url,
)
from app.services.pipeline import FrameFeatures, extract_features, preprocess_frame


@dataclass
class DemoResult:
    source_type: str
    frames: int
    avg_intensity: float
    avg_motion: float
    crowd_level: str
    anomaly_score: float
    anomaly_label: str
    behavior_label: str
    detected_objects: int
    objects: list[dict[str, object]]
    time_series: list[dict[str, object]]
    annotated_frame_data_url: str
    step_metrics: list[dict[str, object]]


def _crowd_level_from_density(density_hint: float) -> str:
    if density_hint < 0.33:
        return "Low"
    if density_hint < 0.66:
        return "Medium"
    return "High"


def _anomaly_from_motion(motion: float) -> tuple[float, str]:
    score = float(np.clip(motion / 4.0, 0.0, 1.0))
    return score, ("Abnormal" if score >= 0.6 else "Normal")


def _behavior_from_signals(avg_motion: float, detected_objects: int, anomaly_label: str) -> str:
    if anomaly_label == "Normal":
        return "normal_flow"
    if detected_objects >= 6 and avg_motion < 0.65:
        return "abnormal_gathering"
    if avg_motion >= 1.8:
        return "fighting"
    if avg_motion >= 1.2:
        return "pushing_shoving"
    if detected_objects <= 2 and avg_motion >= 0.75:
        return "vandalism"
    return "abnormal_gathering"


def _objects_to_dict(items) -> list[dict[str, object]]:
    return [
        {
            "id": obj.object_id,
            "label": obj.label,
            "bbox": [int(v) for v in obj.bbox],
            "score": float(obj.score),
        }
        for obj in items
    ]


def _as_ms(seconds: float) -> float:
    return round(seconds * 1000.0, 2)


def _augment_preview(frame: np.ndarray) -> list[np.ndarray]:
    flips = cv2.flip(frame, 1)
    bright = cv2.convertScaleAbs(frame, alpha=1.12, beta=14)
    noisy = frame.copy()
    noise = np.random.normal(0, 6, frame.shape).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return [flips, bright, noisy]


def _build_metrics(
    source_desc: str,
    source_frames: int,
    augmentation_count: int,
    preprocess_frames: int,
    tracked_objects: int,
    avg_motion: float,
    anomaly_score: float,
    anomaly_label: str,
    behavior_label: str,
    crowd_level: str,
    timings_ms: dict[str, float],
) -> list[dict[str, object]]:
    alert = "ON" if anomaly_label == "Abnormal" else "OFF"
    return [
        {
            "step": "Buoc 1: Input (Anh/Video)",
            "metrics": {
                "source": source_desc,
                "frames_loaded": source_frames,
                "time_ms": timings_ms.get("input", 0.0),
            },
        },
        {
            "step": "Buoc 2: Tang cuong du lieu",
            "metrics": {
                "augmented_samples": augmentation_count,
                "ops": "flip, brightness, noise",
                "time_ms": timings_ms.get("augment", 0.0),
            },
        },
        {
            "step": "Buoc 3: Tien xu ly (1)",
            "metrics": {
                "frames_preprocessed": preprocess_frames,
                "ops": "denoise + equalize",
                "time_ms": timings_ms.get("preprocess_1", 0.0),
            },
        },
        {
            "step": "Buoc 4: Tien xu ly (2)",
            "metrics": {
                "objects_tracked": tracked_objects,
                "avg_motion": round(avg_motion, 4),
                "time_ms": timings_ms.get("preprocess_2", 0.0),
            },
        },
        {
            "step": "Buoc 5: Training & Inference",
            "metrics": {
                "crowd_level": crowd_level,
                "anomaly_score": round(anomaly_score, 4),
                "behavior_label": behavior_label,
                "dynamic_threshold": round(0.5 + min(0.3, avg_motion / 20.0), 4),
                "time_ms": timings_ms.get("inference", 0.0),
            },
        },
        {
            "step": "Buoc 6: Output",
            "metrics": {
                "anomaly_label": anomaly_label,
                "behavior_label": behavior_label,
                "alert": alert,
                "people_flow_objects": tracked_objects,
                "time_ms": timings_ms.get("output", 0.0),
            },
        },
    ]


def _build_result(
    source_type: str,
    features: list[FrameFeatures],
    tracked_objects,
    annotated_frame: np.ndarray,
    step_metrics: list[dict[str, object]],
    behavior_model: BehaviorModelRuntime | BehaviorModelFallback | None = None,
) -> DemoResult:
    if features:
        avg_intensity = float(np.mean([f.mean_intensity for f in features]))
        avg_motion = float(np.mean([f.motion_score for f in features]))
    else:
        avg_intensity = 0.0
        avg_motion = 0.0

    density_hint = float(np.clip(avg_motion / 6.0, 0.0, 1.0))
    anomaly_score, anomaly_label = _anomaly_from_motion(avg_motion)
    objects = _objects_to_dict(tracked_objects)
    behavior_label = _behavior_from_signals(avg_motion, len(objects), anomaly_label)
    if behavior_model is not None and behavior_model.available:
        pred = behavior_model.predict(features)
        behavior_label = pred.label
        if behavior_label != "normal_flow":
            anomaly_label = "Abnormal"
            anomaly_score = max(anomaly_score, pred.confidence)

    if behavior_label not in BEHAVIOR_LABELS:
        behavior_label = "normal_flow"

    return DemoResult(
        source_type=source_type,
        frames=max(1, len(features)),
        avg_intensity=avg_intensity,
        avg_motion=avg_motion,
        crowd_level=_crowd_level_from_density(density_hint),
        anomaly_score=anomaly_score,
        anomaly_label=anomaly_label,
        behavior_label=behavior_label,
        detected_objects=len(objects),
        objects=objects,
        time_series=[],
        annotated_frame_data_url=encode_jpg_data_url(annotated_frame),
        step_metrics=step_metrics,
    )


def analyze_image_frame(
    frame: np.ndarray,
    detector: ObjectDetector,
    tracker: SimpleObjectTracker,
    behavior_model: BehaviorModelRuntime | BehaviorModelFallback | None = None,
) -> DemoResult:
    timings: dict[str, float] = {}

    t0 = perf_counter()
    source = frame.copy()
    timings["input"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    augmented = _augment_preview(source)
    timings["augment"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    processed = preprocess_frame(source)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    feature = FrameFeatures(mean_intensity=float(np.mean(gray)), motion_score=0.0)
    timings["preprocess_1"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    detections = detector.detect_people(source)
    tracked_objects = tracker.update(detections)
    annotated = draw_tracked_objects(source, tracked_objects)
    timings["preprocess_2"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    result = _build_result("image", [feature], tracked_objects, annotated, step_metrics=[], behavior_model=behavior_model)
    result.anomaly_score = 0.0
    result.anomaly_label = "Normal"
    result.behavior_label = "normal_flow"
    timings["inference"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    result.step_metrics = _build_metrics(
        source_desc="image_upload",
        source_frames=1,
        augmentation_count=len(augmented),
        preprocess_frames=1,
        tracked_objects=result.detected_objects,
        avg_motion=result.avg_motion,
        anomaly_score=result.anomaly_score,
        anomaly_label=result.anomaly_label,
        behavior_label=result.behavior_label,
        crowd_level=result.crowd_level,
        timings_ms=timings,
    )
    timings["output"] = _as_ms(perf_counter() - t0)
    result.step_metrics[-1]["metrics"]["time_ms"] = timings["output"]
    return result


def analyze_frame_pair(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    detector: ObjectDetector,
    tracker: SimpleObjectTracker,
    behavior_model: BehaviorModelRuntime | BehaviorModelFallback | None = None,
) -> DemoResult:
    timings: dict[str, float] = {}

    t0 = perf_counter()
    prev = prev_frame.copy()
    cur = cur_frame.copy()
    timings["input"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    augmented = _augment_preview(cur)
    timings["augment"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    prev_processed = preprocess_frame(prev)
    cur_processed = preprocess_frame(cur)
    feature = extract_features(prev_processed, cur_processed)
    timings["preprocess_1"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    detections = detector.detect_people(cur)
    detections.extend(detector.detect_motion(prev, cur))
    tracked_objects = tracker.update(detections)
    annotated = draw_tracked_objects(cur, tracked_objects)
    timings["preprocess_2"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    result = _build_result("webcam", [feature], tracked_objects, annotated, step_metrics=[], behavior_model=behavior_model)
    timings["inference"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    result.step_metrics = _build_metrics(
        source_desc="webcam_stream",
        source_frames=2,
        augmentation_count=len(augmented),
        preprocess_frames=2,
        tracked_objects=result.detected_objects,
        avg_motion=result.avg_motion,
        anomaly_score=result.anomaly_score,
        anomaly_label=result.anomaly_label,
        behavior_label=result.behavior_label,
        crowd_level=result.crowd_level,
        timings_ms=timings,
    )
    timings["output"] = _as_ms(perf_counter() - t0)
    result.step_metrics[-1]["metrics"]["time_ms"] = timings["output"]
    return result


def analyze_video_file(
    video_path: Path,
    detector: ObjectDetector,
    tracker: SimpleObjectTracker,
    behavior_model: BehaviorModelRuntime | BehaviorModelFallback | None = None,
    max_frames: int = 120,
) -> DemoResult:
    timings: dict[str, float] = {}

    t0 = perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    ok, prev = cap.read()
    timings["input"] = _as_ms(perf_counter() - t0)

    if not ok:
        cap.release()
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        empty = _build_result("video", [], [], blank, step_metrics=[], behavior_model=behavior_model)
        empty.step_metrics = _build_metrics(
            source_desc="video_upload",
            source_frames=0,
            augmentation_count=0,
            preprocess_frames=0,
            tracked_objects=0,
            avg_motion=0.0,
            anomaly_score=0.0,
            anomaly_label="NoData",
            behavior_label="normal_flow",
            crowd_level="Unknown",
            timings_ms={
                "input": timings["input"],
                "augment": 0.0,
                "preprocess_1": 0.0,
                "preprocess_2": 0.0,
                "inference": 0.0,
                "output": 0.0,
            },
        )
        return empty

    t0 = perf_counter()
    _augment_preview(prev)
    timings["augment"] = _as_ms(perf_counter() - t0)

    features: list[FrameFeatures] = []
    prev_processed = preprocess_frame(prev)
    last_frame = prev.copy()
    tracked_objects = tracker.update(detector.detect_people(prev))
    frames_read = 1
    time_series: list[dict[str, object]] = []

    preprocess_1_total = 0.0
    preprocess_2_total = 0.0
    frame_count = 0

    while frame_count < max_frames:
        ok, cur = cap.read()
        if not ok:
            break
        frames_read += 1

        t1 = perf_counter()
        cur_processed = preprocess_frame(cur)
        features.append(extract_features(prev_processed, cur_processed))
        preprocess_1_total += perf_counter() - t1

        t2 = perf_counter()
        detections = detector.detect_people(cur)
        detections.extend(detector.detect_motion(prev, cur))
        tracked_objects = tracker.update(detections)
        preprocess_2_total += perf_counter() - t2

        current_behavior = _behavior_from_signals(
            avg_motion=features[-1].motion_score,
            detected_objects=len(tracked_objects),
            anomaly_label=_anomaly_from_motion(features[-1].motion_score)[1],
        )
        if behavior_model is not None and behavior_model.available:
            current_behavior = behavior_model.predict([features[-1]]).label

        time_series.append(
            {
                "frame_index": frames_read - 1,
                "t_sec": round((frames_read - 1) / fps, 3),
                "motion": round(features[-1].motion_score, 4),
                "mean_intensity": round(features[-1].mean_intensity, 4),
                "object_count": len(tracked_objects),
                "behavior_label": current_behavior,
            }
        )

        last_frame = cur.copy()
        prev = cur
        prev_processed = cur_processed
        frame_count += 1

    timings["preprocess_1"] = _as_ms(preprocess_1_total)
    timings["preprocess_2"] = _as_ms(preprocess_2_total)

    cap.release()

    t0 = perf_counter()
    annotated = draw_tracked_objects(last_frame, tracked_objects)
    result = _build_result("video", features, tracked_objects, annotated, step_metrics=[], behavior_model=behavior_model)
    timings["inference"] = _as_ms(perf_counter() - t0)

    t0 = perf_counter()
    result.step_metrics = _build_metrics(
        source_desc="video_upload",
        source_frames=frames_read,
        augmentation_count=3,
        preprocess_frames=frames_read,
        tracked_objects=result.detected_objects,
        avg_motion=result.avg_motion,
        anomaly_score=result.anomaly_score,
        anomaly_label=result.anomaly_label,
        behavior_label=result.behavior_label,
        crowd_level=result.crowd_level,
        timings_ms=timings,
    )
    timings["output"] = _as_ms(perf_counter() - t0)
    result.step_metrics[-1]["metrics"]["time_ms"] = timings["output"]
    result.time_series = time_series

    return result
