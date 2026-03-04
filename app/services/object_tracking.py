"""Object detection and simple centroid-based tracking for demo."""

from __future__ import annotations

from dataclasses import dataclass
import base64

import cv2
import numpy as np


@dataclass
class DetectedObject:
    object_id: int
    label: str
    bbox: tuple[int, int, int, int]
    score: float


class SimpleObjectTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 12):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, dict[str, object]] = {}

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    def update(self, detections: list[dict[str, object]]) -> list[DetectedObject]:
        assigned_track_ids: set[int] = set()
        tracked: list[DetectedObject] = []

        for det in detections:
            bbox = det["bbox"]
            center = self._center(bbox)
            label = str(det.get("label", "Object"))
            score = float(det.get("score", 0.0))

            best_id = None
            best_distance = None
            for track_id, track in self.tracks.items():
                if track_id in assigned_track_ids:
                    continue
                tx, ty = track["center"]
                distance = float(np.hypot(center[0] - tx, center[1] - ty))
                if distance <= self.max_distance and (best_distance is None or distance < best_distance):
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {
                "center": center,
                "label": label,
                "bbox": bbox,
                "score": score,
                "missed": 0,
            }
            assigned_track_ids.add(best_id)
            tracked.append(DetectedObject(best_id, label, bbox, score))

        stale_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id not in assigned_track_ids:
                track["missed"] = int(track["missed"]) + 1
                if int(track["missed"]) > self.max_missed:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracks[track_id]

        tracked.sort(key=lambda item: item.object_id)
        return tracked


class ObjectDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_people(self, frame: np.ndarray) -> list[dict[str, object]]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections: list[dict[str, object]] = []
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "label": "Person",
                    "score": confidence,
                }
            )
        return detections

    @staticmethod
    def detect_motion(prev_frame: np.ndarray, cur_frame: np.ndarray) -> list[dict[str, object]]:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, cur_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blur, 22, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, object]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1200:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "label": "MovingObject",
                    "score": min(1.0, float(area / 12000.0)),
                }
            )
        return detections


def draw_tracked_objects(frame: np.ndarray, objects: list[DetectedObject]) -> np.ndarray:
    canvas = frame.copy()
    for obj in objects:
        x, y, w, h = obj.bbox
        color = (40, 180, 250) if obj.label == "Person" else (0, 220, 120)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        tag = f"{obj.label} #{obj.object_id}"
        cv2.putText(canvas, tag, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    return canvas


def encode_jpg_data_url(frame: np.ndarray, quality: int = 80) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""

    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
