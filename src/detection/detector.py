"""
Branch 2 — Object Detection & Segmentation
==========================================
  • PersonDetector: YOLOv8-based person detection
  • BackgroundSubtractor: MOG2/KNN motion detection (no GPU needed)
  • DensityEstimator: crowd density heatmap via Gaussian blobs
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single person detection result."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


# ── YOLO-based Person Detector ────────────────────────────────────────────

class PersonDetector:
    """
    Detects persons in images using YOLOv8.

    Falls back to a lightweight HOG detector if ultralytics is unavailable
    so the project runs on plain CPU without downloading weights.

    Usage:
        detector = PersonDetector(cfg["detection"])
        detections = detector.detect(frame)
    """

    def __init__(self, cfg: dict):
        self.conf  = cfg.get("confidence", 0.45)
        self.iou   = cfg.get("iou_threshold", 0.45)
        self.size  = cfg.get("input_size", 640)
        self.model = None
        self._backend = "none"
        self._init_model(cfg.get("model", "yolov8n"))

    def _init_model(self, model_name: str):
        """Try YOLO first, fall back to HOG."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{model_name}.pt")
            self._backend = "yolo"
            logger.info(f"PersonDetector: loaded {model_name} via Ultralytics")
        except Exception as e:
            logger.warning(f"YOLO unavailable ({e}), using HOG detector as fallback")
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._backend = "hog"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a BGR frame.

        Returns:
            List of Detection objects sorted by confidence descending.
        """
        if self._backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],          # person only
            imgsz=self.size,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _detect_hog(self, frame: np.ndarray) -> List[Detection]:
        """HOG-based fallback – works without GPU or YOLO weights."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = self.model.detectMultiScale(
            gray, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        detections = []
        for (x, y, w, h), conf in zip(rects, weights):
            if conf > 0.3:
                detections.append(Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=float(conf[0]),
                ))
        return detections


# ── Background Subtractor ─────────────────────────────────────────────────

class MotionDetector:
    """
    Background subtraction for motion detection.
    Useful for detecting any movement before running heavy YOLO inference.

    Usage:
        motion = MotionDetector(method="mog2")
        mask, contours = motion.detect(frame)
    """

    def __init__(self, method: str = "mog2", min_area: int = 500):
        self.min_area = min_area
        if method.lower() == "knn":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400, detectShadows=True
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        logger.info(f"MotionDetector: backend={method}")

    def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Apply background subtraction and find motion blobs.

        Returns:
            (fg_mask, bounding_boxes) where boxes are (x, y, w, h)
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # Remove shadows (grey pixels → 0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
        return fg_mask, boxes


# ── Crowd Density Estimator ───────────────────────────────────────────────

class DensityEstimator:
    """
    Lightweight crowd density estimator.

    Places a Gaussian blob at each detection center and accumulates
    over time to build a density/heatmap.

    Usage:
        density = DensityEstimator(frame_shape=(480, 640))
        density.update(detections)
        heatmap = density.get_heatmap()
        count = density.count_in_zone(zone_rect)
    """

    def __init__(self, frame_shape: Tuple[int, int], decay: float = 0.97, radius: int = 20):
        h, w = frame_shape[:2]
        self.heatmap = np.zeros((h, w), dtype=np.float32)
        self.decay = decay
        self.radius = radius
        self._sigma = radius / 3

    def update(self, detections: List[Detection]) -> None:
        """Add detection centers to accumulated heatmap."""
        self.heatmap *= self.decay
        for det in detections:
            cx, cy = det.center
            if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
                cv2.circle(self.heatmap, (cx, cy), self.radius, 1.0, -1)
        self.heatmap = cv2.GaussianBlur(self.heatmap, (0, 0), self._sigma)

    def get_heatmap(self) -> np.ndarray:
        """Return current float32 heatmap (H, W)."""
        return self.heatmap.copy()

    def count_in_zone(self, zone: Tuple[int, int, int, int], detections: List[Detection]) -> int:
        """Count detections whose center falls inside zone (x1,y1,x2,y2)."""
        x1, y1, x2, y2 = zone
        return sum(
            1 for d in detections
            if x1 <= d.center[0] <= x2 and y1 <= d.center[1] <= y2
        )

    def reset(self) -> None:
        self.heatmap[:] = 0
