"""Pipeline helpers for video preprocessing and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    mean_intensity: float
    motion_score: float


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Denoise and equalize luminance."""
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


def _resize_if_needed(frame: np.ndarray, resize_width: int | None) -> np.ndarray:
    if not resize_width or resize_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame
    scale = resize_width / float(w)
    new_size = (resize_width, max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def process_video(
    video_path: Path,
    max_frames: int = 120,
    frame_stride: int = 1,
    resize_width: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[FrameFeatures]:
    features, _ = process_video_with_processed_frames(
        video_path=video_path,
        max_frames=max_frames,
        frame_stride=frame_stride,
        resize_width=resize_width,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return features


def process_video_with_processed_frames(
    video_path: Path,
    max_frames: int = 120,
    frame_stride: int = 1,
    resize_width: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[list[FrameFeatures], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    features: list[FrameFeatures] = []
    processed_frames: list[np.ndarray] = []
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return features, processed_frames

    prev = _resize_if_needed(prev, resize_width)
    prev = preprocess_frame(prev)
    processed_frames.append(prev.copy())
    frame_count = 0
    stride = max(1, int(frame_stride))

    while frame_count < max_frames:
        if end_frame is not None:
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_pos >= int(end_frame):
                break

        ok, cur = cap.read()
        if not ok:
            break

        if stride > 1:
            for _ in range(stride - 1):
                ok = cap.grab()
                if not ok:
                    break
            if not ok:
                break

        cur = _resize_if_needed(cur, resize_width)
        cur = preprocess_frame(cur)
        features.append(extract_features(prev, cur))
        processed_frames.append(cur.copy())
        prev = cur
        frame_count += 1

    cap.release()
    return features, processed_frames
