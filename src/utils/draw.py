"""src/utils/draw.py — Tất cả hàm vẽ lên frame."""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# Màu per-class (BGR)
COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 150),    # Normal     – teal
    1: (0,  50, 230),    # Fighting   – đỏ
    2: (0, 165, 255),    # Falling    – cam
    3: (200, 50, 200),   # Loitering  – tím
    4: (0,   0, 220),    # Crowd Panic– đỏ đậm
}
NAMES = {0: "Normal", 1: "Fighting", 2: "Falling", 3: "Loitering", 4: "Panic"}


def boxes(
    frame: np.ndarray,
    bboxes: List[Tuple[int,int,int,int]],
    track_ids: Optional[List[int]] = None,
    labels:    Optional[List[int]] = None,
    confs:     Optional[List[float]] = None,
) -> np.ndarray:
    """Vẽ bounding-box + nhãn lên frame."""
    out = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cls   = labels[i]    if labels    else 0
        color = COLORS.get(cls, (100, 200, 100))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        parts: List[str] = []
        if track_ids: parts.append(f"#{track_ids[i]}")
        if labels:    parts.append(NAMES.get(cls, str(cls)))
        if confs:     parts.append(f"{confs[i]:.2f}")
        txt = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return out


def trajectories(
    frame: np.ndarray,
    traj: Dict[int, List[Tuple[int, int]]],
    max_pts: int = 35,
) -> np.ndarray:
    out = frame.copy()
    for tid, pts in traj.items():
        trail = pts[-max_pts:]
        color = COLORS.get(tid % 5, (200, 200, 0))
        for j in range(1, len(trail)):
            alpha = j / len(trail)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(out, trail[j-1], trail[j], c, 2)
        if trail:
            cv2.circle(out, trail[-1], 4, color, -1)
    return out


def heatmap(
    frame: np.ndarray,
    acc: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay heatmap mật độ lên frame."""
    if acc.max() < 1e-6:
        return frame.copy()
    norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(
        cv2.resize(norm, (frame.shape[1], frame.shape[0])),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)


def stats_panel(
    frame: np.ndarray,
    stats: Dict[str, str],
    x: int = 8, y: int = 24,
    line_h: int = 22,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    ph = len(stats) * line_h + 10
    cv2.rectangle(overlay, (x-4, y-20), (x+210, y+ph-20), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    for i, (k, v) in enumerate(stats.items()):
        cv2.putText(out, f"{k}: {v}", (x, y + i*line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
    return out


def alert_banner(frame: np.ndarray, text: str, color=(0,0,220)) -> np.ndarray:
    """Dải cảnh báo đỏ chớp lên trên cùng."""
    out = frame.copy()
    h = 38
    cv2.rectangle(out, (0,0), (out.shape[1], h), color, -1)
    cv2.putText(out, f"⚠  {text}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out
