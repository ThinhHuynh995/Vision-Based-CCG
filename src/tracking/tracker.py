"""
Branch 5 — Video Processing & Object Tracking
=============================================
  • SimpleTracker: IoU-based multi-object tracker (no heavy deps)
  • TrackState: per-track state with velocity & loitering detection
  • OpticalFlowAnalyzer: Farneback dense optical flow
  • VideoProcessor: orchestrates detection + tracking + anomaly on video files
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from src.detection.detector import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Track State ───────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Stores per-object tracking history and computes anomaly signals."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    label: int = 0
    confidence: float = 1.0

    # History
    centers: List[Tuple[int, int]] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    frames_stationary: int = 0
    frames_alive: int = 0
    last_seen: int = 0

    def update(self, bbox, center, label, confidence, frame_idx: int):
        self.bbox = bbox
        # Velocity
        if self.centers:
            dx = center[0] - self.centers[-1][0]
            dy = center[1] - self.centers[-1][1]
            v = np.sqrt(dx ** 2 + dy ** 2)
            self.velocities.append(v)
            if v < 3:
                self.frames_stationary += 1
            else:
                self.frames_stationary = 0
        self.centers.append(center)
        self.center = center
        self.label = label
        self.confidence = confidence
        self.frames_alive += 1
        self.last_seen = frame_idx
        # Keep history bounded
        if len(self.centers) > 120:
            self.centers.pop(0)
        if len(self.velocities) > 120:
            self.velocities.pop(0)

    @property
    def avg_speed(self) -> float:
        if not self.velocities:
            return 0.0
        return float(np.mean(self.velocities[-10:]))

    @property
    def is_loitering(self) -> bool:
        return self.frames_stationary >= 60

    @property
    def is_speeding(self) -> bool:
        return self.avg_speed > 80

    def trajectory(self) -> List[Tuple[int, int]]:
        return self.centers[-40:]


# ── IoU-based Simple Tracker ──────────────────────────────────────────────

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class SimpleTracker:
    """
    Lightweight IoU-based multi-object tracker.

    No external dependencies. Works purely on bounding box overlap.
    Suitable for CPU-only environments.

    Usage:
        tracker = SimpleTracker(cfg["tracking"])
        tracks = tracker.update(detections, labels, frame_idx)
    """

    def __init__(self, cfg: dict):
        self.iou_threshold = cfg.get("iou_threshold", 0.3)
        self.max_age = cfg.get("max_age", 30)
        self.min_hits = cfg.get("min_hits", 3)
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        labels: Optional[List[int]] = None,
        confidences: Optional[List[float]] = None,
        frame_idx: int = 0,
    ) -> List[TrackState]:
        """
        Match current detections to existing tracks via IoU.

        Args:
            detections: list of Detection objects
            labels: optional behavior label per detection
            confidences: optional confidence per detection
            frame_idx: current frame number

        Returns:
            List of active TrackState objects
        """
        if labels is None:
            labels = [0] * len(detections)
        if confidences is None:
            confidences = [1.0] * len(detections)

        matched = set()
        used_tracks = set()

        # Match detections to existing tracks
        for i, det in enumerate(detections):
            best_iou, best_tid = 0.0, None
            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou >= self.iou_threshold and best_tid is not None:
                self.tracks[best_tid].update(
                    det.bbox, det.center, labels[i], confidences[i], frame_idx
                )
                matched.add(i)
                used_tracks.add(best_tid)
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                ts = TrackState(
                    track_id=tid,
                    bbox=det.bbox,
                    center=det.center,
                    label=labels[i],
                    confidence=confidences[i],
                )
                ts.centers.append(det.center)
                ts.last_seen = frame_idx
                self.tracks[tid] = ts

        # Remove stale tracks
        stale = [tid for tid, t in self.tracks.items()
                 if frame_idx - t.last_seen > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return [t for t in self.tracks.values() if t.frames_alive >= self.min_hits]

    def get_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        return {tid: t.trajectory() for tid, t in self.tracks.items()}

    def reset(self):
        self.tracks.clear()
        self._next_id = 1


# ── Optical Flow ──────────────────────────────────────────────────────────

class OpticalFlowAnalyzer:
    """
    Dense optical flow using Farneback algorithm.

    Visualizes motion direction & magnitude on video frames.

    Usage:
        flow = OpticalFlowAnalyzer()
        magnitude_map, flow_vis = flow.analyze(prev_frame, curr_frame)
    """

    def analyze(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow between two consecutive frames.

        Returns:
            (magnitude_map, visualization_bgr)
        """
        g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # HSV visualization
        hsv = np.zeros((*g1.shape, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return mag, vis

    def mean_motion(self, magnitude_map: np.ndarray) -> float:
        """Average motion magnitude across frame."""
        return float(magnitude_map.mean())


# ── Anomaly Detector ──────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Rule-based anomaly detector that combines:
      - Behavior classification label
      - Track speed
      - Loitering duration
      - Crowd density
    """

    def __init__(self, cfg: dict):
        self.speed_thresh   = cfg.get("speed_threshold", 80)
        self.loiter_frames  = cfg.get("loiter_frames", 60)
        self.crowd_thresh   = cfg.get("crowd_density_threshold", 5)
        self.anomalies: List[Dict] = []

    def check_track(self, track: TrackState, frame_idx: int) -> Optional[str]:
        """
        Evaluate a single track for anomalies.

        Returns:
            Anomaly type string or None if normal
        """
        # DL classifier flagged abnormal
        if track.label in {1, 2, 4}:  # Fighting, Falling, Panic
            return {1: "FIGHTING", 2: "FALLING", 4: "CROWD PANIC"}[track.label]
        # Rule: too fast
        if track.is_speeding:
            return "RUNNING FAST"
        # Rule: loitering
        if track.label == 3 or track.is_loitering:
            return "LOITERING"
        return None

    def check_crowd(self, count: int) -> Optional[str]:
        """Trigger crowd alert if density exceeds threshold."""
        if count >= self.crowd_thresh:
            return f"CROWD DENSITY ({count})"
        return None

    def log_anomaly(self, anomaly_type: str, track_id: int, frame_idx: int, center: Tuple):
        entry = {
            "frame": frame_idx,
            "track_id": track_id,
            "type": anomaly_type,
            "center": center,
            "time": time.strftime("%H:%M:%S"),
        }
        self.anomalies.append(entry)
        logger.warning(f"⚠  ANOMALY [{anomaly_type}] track={track_id} frame={frame_idx}")

    def get_report(self) -> List[Dict]:
        return self.anomalies


# ── Video Processor ───────────────────────────────────────────────────────

class VideoProcessor:
    """
    Full video processing pipeline:
    preprocess → detect → classify → track → anomaly detection

    Usage:
        vp = VideoProcessor(cfg)
        vp.process_video("input.mp4", "output.mp4")
    """

    def __init__(self, cfg: dict):
        from src.preprocessing.image_processor import ImagePreprocessor
        from src.detection.detector import PersonDetector, DensityEstimator
        from src.classification.behavior_classifier import BehaviorClassifier

        self.cfg = cfg
        self.preprocessor   = ImagePreprocessor(cfg.get("preprocessing", {}))
        self.detector       = PersonDetector(cfg.get("detection", {}))
        self.classifier     = BehaviorClassifier(cfg.get("classification", {}))
        self.tracker        = SimpleTracker(cfg.get("tracking", {}))
        self.anomaly_det    = AnomalyDetector(cfg.get("anomaly", {}))
        self.flow_analyzer  = OpticalFlowAnalyzer() if cfg.get("tracking", {}).get("optical_flow", True) else None

        self._density: Optional[DensityEstimator] = None
        logger.info("VideoProcessor initialized")

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_heatmap: bool = True,
        show_flow: bool = False,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Process a video file end-to-end.

        Args:
            input_path:   path to input video / webcam index (int)
            output_path:  path to save annotated video
            show_heatmap: overlay crowd density heatmap
            show_flow:    overlay optical flow visualization
            max_frames:   limit processing to N frames (None = all)

        Returns:
            Summary dict with frame count, anomaly log, timing
        """
        from src.utils.visualizer import (
            draw_detections, draw_trajectories, build_heatmap, put_stats
        )

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from pathlib import Path as _P
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self._density = __import__(
            "src.detection.detector", fromlist=["DensityEstimator"]
        ).DensityEstimator(
            (h, w),
            decay=self.cfg.get("tracking", {}).get("heatmap_decay", 0.97),
            radius=self.cfg.get("tracking", {}).get("density_radius", 20),
        )

        prev_frame = None
        frame_idx = 0
        t0 = time.time()
        anomaly_count = 0

        logger.info(f"Processing {total} frames from {input_path}")

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            # ── Branch 1: Preprocess ──────────────────────
            clean = self.preprocessor.process(frame)

            # ── Branch 2: Detect ──────────────────────────
            detections = self.detector.detect(clean)
            self._density.update(detections)

            # ── Branch 3: Classify crops ──────────────────
            labels, confs = [], []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                crop = clean[max(0,y1):y2, max(0,x1):x2]
                if crop.size > 0:
                    lbl, conf = self.classifier.predict(crop)
                else:
                    lbl, conf = 0, 1.0
                labels.append(lbl)
                confs.append(conf)

            # ── Branch 5: Track ───────────────────────────
            tracks = self.tracker.update(detections, labels, confs, frame_idx)

            # ── Anomaly Check ─────────────────────────────
            for track in tracks:
                anomaly = self.anomaly_det.check_track(track, frame_idx)
                if anomaly:
                    anomaly_count += 1
                    self.anomaly_det.log_anomaly(anomaly, track.track_id, frame_idx, track.center)

            crowd_alert = self.anomaly_det.check_crowd(len(detections))

            # ── Draw ──────────────────────────────────────
            out = frame.copy()
            boxes  = [d.bbox  for d in detections]
            tids   = [None]    # track IDs populated below
            tids   = [t.track_id for t in tracks] if tracks else None
            t_boxes = [t.bbox for t in tracks]
            t_lbls  = [t.label for t in tracks]
            t_confs = [t.confidence for t in tracks]

            if show_heatmap:
                out = build_heatmap(out, self._density.get_heatmap())

            if t_boxes:
                out = draw_detections(out, t_boxes, tids, t_lbls, t_confs)

            traj = self.tracker.get_trajectories()
            out = draw_trajectories(out, traj)

            if show_flow and prev_frame is not None and self.flow_analyzer:
                _, flow_vis = self.flow_analyzer.analyze(prev_frame, clean)
                flow_small = cv2.resize(flow_vis, (w // 4, h // 4))
                out[0:h // 4, 0:w // 4] = flow_small

            # Stats panel
            stats = {
                "Frame":   str(frame_idx),
                "Persons": str(len(detections)),
                "Tracks":  str(len(tracks)),
                "Alerts":  str(anomaly_count),
            }
            if crowd_alert:
                stats["⚠ CROWD"] = crowd_alert
            out = put_stats(out, stats)

            writer.write(out)
            prev_frame = clean.copy()
            frame_idx += 1

        cap.release()
        writer.release()
        elapsed = time.time() - t0

        summary = {
            "frames_processed": frame_idx,
            "total_anomalies": anomaly_count,
            "anomaly_log": self.anomaly_det.get_report(),
            "elapsed_sec": round(elapsed, 2),
            "fps_achieved": round(frame_idx / max(elapsed, 1), 1),
        }
        logger.info(
            f"Done. {frame_idx} frames in {elapsed:.1f}s "
            f"({summary['fps_achieved']} fps) | {anomaly_count} anomalies"
        )
        return summary
