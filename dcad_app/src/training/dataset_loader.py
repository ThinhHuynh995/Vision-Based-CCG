from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np


@dataclass
class VideoEntry:
    path: Path
    label: int
    folder: str
    reason: str


@dataclass
class DatasetScan:
    root: Path
    videos: list[VideoEntry]
    has_masks: bool


@dataclass
class MultiDatasetScan:
    videos: list[VideoEntry]


class DatasetLoader:
    NORMAL_KEYWORDS = ["normal", "train", "training", "background", "bình_thường", "binh_thuong", "negative"]
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg"}

    def is_normal(self, folder_name: str) -> bool:
        low = folder_name.lower()
        return any(kw in low for kw in self.NORMAL_KEYWORDS)

    def scan_dataset(self, root: Path) -> DatasetScan:
        videos: list[VideoEntry] = []
        for folder in root.iterdir() if root.exists() else []:
            if not folder.is_dir():
                continue
            label = 0 if self.is_normal(folder.name) else 1
            reason = "keyword" if label == 0 else "non-normal"
            for f in folder.rglob("*"):
                if f.is_file() and f.suffix.lower() in self.VIDEO_EXTS:
                    videos.append(VideoEntry(path=f, label=label, folder=folder.name, reason=reason))
        has_masks = (root / "test_frame_mask").exists() or any(p.name == "frame_mask" for p in root.rglob("*") if p.is_dir())
        return DatasetScan(root=root, videos=videos, has_masks=has_masks)

    def scan_multiple(self, dataset_configs: list[dict]) -> MultiDatasetScan:
        all_videos = []
        for cfg in dataset_configs:
            scan = self.scan_dataset(Path(cfg["path"]))
            for v in scan.videos:
                if cfg.get("override_label") is not None:
                    v.label = int(cfg["override_label"])
                all_videos.append(v)
        return MultiDatasetScan(videos=all_videos)

    def iter_clips(self, video_path: Path, fps: int = 5, resize: int = 64, clip_len: int = 16, max_clips: int = 20, augment: bool = False):
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(src_fps / fps))
        frames = []
        idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (resize, resize)))
            idx += step
        cap.release()

        clips = []
        stride = max(1, clip_len // 2)
        for start in range(0, len(frames) - clip_len + 1, stride):
            clips.append(np.array(frames[start : start + clip_len], dtype=np.uint8))
            if len(clips) >= max_clips:
                break

        if augment:
            aug = []
            for clip in clips:
                c = clip.copy()
                if random.random() > 0.5:
                    c = c[:, :, ::-1, :]
                if random.random() > 0.5:
                    noise = np.random.normal(0, 10, c.shape).astype(np.int16)
                    c = np.clip(c.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                aug.append(c)
            clips.extend(aug)
        return clips

    def load_shanghaitech_masks(self, mask_dir: Path) -> dict:
        masks = {}
        for m in mask_dir.rglob("*.npy"):
            masks[m.stem] = np.load(str(m))
        for m in mask_dir.rglob("*.png"):
            img = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks[m.stem] = (img > 127).astype(np.uint8)
        return masks
