"""
Utility: Drawing helpers – bounding boxes, labels, heatmaps, trajectories.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


# ── Color map per behavior class ──────────────────────────────────────────
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 150),    # Normal    – teal
    1: (0,  60, 220),    # Fighting  – red (BGR)
    2: (0, 165, 255),    # Falling   – orange
    3: (200, 50, 200),   # Loitering – purple
    4: (0,   0, 220),    # Panic     – red
}
CLASS_NAMES = {0: "Normal", 1: "Fighting", 2: "Falling", 3: "Loitering", 4: "Panic"}


def draw_detections(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    track_ids: Optional[List[int]] = None,
    labels: Optional[List[int]] = None,
    confidences: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes with optional track IDs and behavior labels.

    Args:
        frame: BGR image (H, W, 3)
        boxes: list of (x1, y1, x2, y2)
        track_ids: list of integer track IDs
        labels: list of class indices
        confidences: list of confidence scores [0,1]

    Returns:
        Annotated frame copy
    """
    out = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cls = labels[i] if labels else 0
        color = CLASS_COLORS.get(cls, (100, 200, 100))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        parts = []
        if track_ids:
            parts.append(f"ID:{track_ids[i]}")
        if labels:
            parts.append(CLASS_NAMES.get(cls, f"cls{cls}"))
        if confidences:
            parts.append(f"{confidences[i]:.2f}")
        text = " | ".join(parts)

        # Background pill for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def draw_trajectories(
    frame: np.ndarray,
    trajectories: Dict[int, List[Tuple[int, int]]],
    max_len: int = 30,
) -> np.ndarray:
    """
    Draw per-track trajectory trails on frame.

    Args:
        frame: BGR image
        trajectories: {track_id: [(cx, cy), ...]} – ordered oldest→newest
        max_len: max number of points to draw per trail

    Returns:
        Annotated frame copy
    """
    out = frame.copy()
    for tid, pts in trajectories.items():
        trail = pts[-max_len:]
        color = CLASS_COLORS.get(tid % len(CLASS_COLORS), (200, 200, 0))
        for j in range(1, len(trail)):
            alpha = j / len(trail)
            c = tuple(int(x * alpha) for x in color)
            cv2.line(out, trail[j - 1], trail[j], c, 2)
        if trail:
            cv2.circle(out, trail[-1], 4, color, -1)
    return out


def build_heatmap(
    frame: np.ndarray,
    heatmap_acc: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay accumulated density heatmap on frame.

    Args:
        frame: BGR image (H, W, 3)
        heatmap_acc: float32 accumulation array (H, W)
        alpha: blend weight for heatmap

    Returns:
        Frame with heatmap overlay
    """
    if heatmap_acc.max() < 1e-6:
        return frame.copy()

    norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)


def put_stats(
    frame: np.ndarray,
    stats: Dict[str, str],
    origin: Tuple[int, int] = (10, 25),
    line_height: int = 22,
) -> np.ndarray:
    """
    Overlay a stats panel (key: value) in top-left corner.

    Args:
        frame: BGR image
        stats: ordered dict of label → value strings
        origin: (x, y) of first line
        line_height: pixels between lines

    Returns:
        Annotated frame copy
    """
    out = frame.copy()
    x, y = origin
    # Semi-transparent background
    overlay = out.copy()
    panel_h = len(stats) * line_height + 10
    cv2.rectangle(overlay, (x - 5, y - 18), (x + 200, y + panel_h - 18), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    for i, (k, v) in enumerate(stats.items()):
        text = f"{k}: {v}"
        cv2.putText(out, text, (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    return out
