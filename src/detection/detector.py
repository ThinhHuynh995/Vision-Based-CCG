"""
src/detection/detector.py — Branch 2: Object Detection & Density
=================================================================
  • PersonDetector  – YOLOv8 (fallback: HOG) với motion gate
  • MotionGate      – MOG2 background subtractor để lọc frame tĩnh
  • DensityMap      – Heatmap mật độ người theo thời gian
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import src.utils.log as _log

log = _log.get(__name__)


@dataclass
class Det:
    """Một kết quả phát hiện người."""
    bbox: Tuple[int,int,int,int]   # (x1,y1,x2,y2)
    conf: float
    center: Tuple[int,int] = field(init=False)

    def __post_init__(self):
        x1,y1,x2,y2 = self.bbox
        self.center = ((x1+x2)//2, (y1+y2)//2)

    @property
    def w(self): return self.bbox[2]-self.bbox[0]

    @property
    def h(self): return self.bbox[3]-self.bbox[1]

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x1,y1,x2,y2 = self.bbox
        return frame[max(0,y1):y2, max(0,x1):x2]


# ── Motion Gate ──────────────────────────────────────────────────────────────

class MotionGate:
    """
    Dùng MOG2 để phát hiện frame có chuyển động.
    Giúp bỏ qua YOLO trên frame tĩnh → tiết kiệm CPU.
    """
    def __init__(self, min_area: int = 800):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.min_area = min_area

    def has_motion(self, frame: np.ndarray) -> bool:
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) >= self.min_area for c in cnts)

    def fg_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self._bg.apply(frame)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)


# ── Person Detector ──────────────────────────────────────────────────────────

class PersonDetector:
    """
    Phát hiện người bằng YOLOv8n.
    Tự động fallback sang HOG nếu ultralytics chưa cài.

    Sử dụng:
        det = PersonDetector(cfg.detection)
        dets = det(frame)          # List[Det]
    """

    def __init__(self, cfg):
        self._conf  = float(cfg.confidence    or 0.45)
        self._iou   = float(cfg.iou_threshold or 0.45)
        self._size  = int(cfg.input_size      or 640)
        self._use_motion = bool(cfg.motion_filter if cfg.motion_filter is not None else True)
        self._model      = None
        self._backend    = "none"
        self._gate       = MotionGate() if self._use_motion else None
        self._init(str(cfg.model or "yolov8n"))

    def _init(self, model_name: str):
        try:
            from ultralytics import YOLO
            self._model   = YOLO(f"{model_name}.pt")
            self._backend = "yolo"
            log.info(f"PersonDetector: YOLO {model_name}")
        except Exception as e:
            log.warning(f"YOLO không khả dụng ({e}) → dùng HOG fallback")
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._model   = hog
            self._backend = "hog"

    def __call__(self, frame: np.ndarray) -> List[Det]:
        if frame is None or frame.size == 0:
            return []
        # Bỏ qua nếu không có chuyển động
        if self._gate and not self._gate.has_motion(frame):
            return []
        return self._yolo(frame) if self._backend == "yolo" else self._hog(frame)

    def _yolo(self, frame: np.ndarray) -> List[Det]:
        results = self._model(
            frame, conf=self._conf, iou=self._iou,
            classes=[0], imgsz=self._size, verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                dets.append(Det(bbox=(x1,y1,x2,y2), conf=float(box.conf[0])))
        return sorted(dets, key=lambda d: d.conf, reverse=True)

    def _hog(self, frame: np.ndarray) -> List[Det]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = self._model.detectMultiScale(
            gray, winStride=(8,8), padding=(4,4), scale=1.05
        )
        dets = []
        for (x,y,w,h), wt in zip(rects, weights):
            if float(wt[0]) > 0.3:
                dets.append(Det(bbox=(x,y,x+w,y+h), conf=float(wt[0])))
        return dets


# ── Density Map ──────────────────────────────────────────────────────────────

class DensityMap:
    """
    Tích lũy heatmap Gaussian tại vị trí trung tâm mỗi người.
    Dùng để visualize điểm tập trung và phát hiện đám đông.

    Sử dụng:
        dm = DensityMap((h, w))
        dm.update(dets)
        heatmap_arr = dm.get()
        count = dm.count_in_zone((x1,y1,x2,y2), dets)
    """

    def __init__(self, shape: Tuple[int,int], decay: float = 0.97, radius: int = 22):
        self._map   = np.zeros(shape[:2], dtype=np.float32)
        self._decay = decay
        self._sigma = radius / 3
        self._r     = radius

    def update(self, dets: List[Det]) -> None:
        self._map *= self._decay
        for d in dets:
            cx, cy = d.center
            h, w   = self._map.shape
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(self._map, (cx, cy), self._r, 1.0, -1)
        self._map = cv2.GaussianBlur(self._map, (0,0), self._sigma)

    def get(self) -> np.ndarray:
        return self._map.copy()

    def count_in_zone(
        self, zone: Tuple[int,int,int,int], dets: List[Det]
    ) -> int:
        x1,y1,x2,y2 = zone
        return sum(1 for d in dets if x1<=d.center[0]<=x2 and y1<=d.center[1]<=y2)

    def reset(self) -> None:
        self._map[:] = 0
