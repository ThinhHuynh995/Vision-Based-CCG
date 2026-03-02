"""Pipeline mô phỏng các bước xử lý video theo đề tài DCAD."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    mean_intensity: float
    motion_score: float


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh: khử nhiễu + cân bằng sáng."""
    denoised = cv2.medianBlur(frame, 5)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def extract_features(prev_frame: np.ndarray, cur_frame: np.ndarray) -> FrameFeatures:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return FrameFeatures(
        mean_intensity=float(np.mean(cur_gray)),
        motion_score=float(np.mean(magnitude)),
    )


def process_video(video_path: Path, max_frames: int = 120) -> list[FrameFeatures]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    features: list[FrameFeatures] = []
    ok, prev = cap.read()
    if not ok:
        return features

    prev = preprocess_frame(prev)
    frame_count = 0

    while frame_count < max_frames:
        ok, cur = cap.read()
        if not ok:
            break
        cur = preprocess_frame(cur)
        features.append(extract_features(prev, cur))
        prev = cur
        frame_count += 1

    cap.release()
    return features
