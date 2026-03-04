from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset_loader import DatasetLoader, MultiDatasetScan


class FeatureExtractor:
    def __init__(self):
        self.loader = DatasetLoader()

    def extract_raw(self, clip: np.ndarray) -> np.ndarray:
        return clip.mean(axis=0).flatten().astype(np.float32)

    def extract_handcrafted(self, clip: np.ndarray) -> np.ndarray:
        mean_frame = clip.mean(axis=0).astype(np.uint8)
        gray = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(sobx**2 + soby**2)
        hsv = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts] or [0.0]
        aspect = []
        for c in cnts[:20]:
            x, y, w, h = cv2.boundingRect(c)
            aspect.append(w / max(h, 1))
        flow_like = np.diff(clip.astype(np.float32), axis=0)
        motion_mag = np.abs(flow_like).mean(axis=(1, 2, 3)) if len(flow_like) else np.zeros(1)
        feat = np.array([
            float(lap.mean()), float(lap.std()), float(np.var(lap)), float(np.percentile(lap, 90)),
            float(mag.mean()), float(mag.std()), float(motion_mag.mean()), float(motion_mag.std()),
            float(np.clip(mag.mean() / 255.0, 0, 1)), float(np.clip(np.count_nonzero(edges) / edges.size, 0, 1)), float(np.clip(len(cnts) / 100, 0, 1)),
            float(hsv[:, :, 0].mean()), float(hsv[:, :, 0].std()), float(hsv[:, :, 1].mean()), float(hsv[:, :, 1].std()), float(hsv[:, :, 2].mean()), float(hsv[:, :, 2].std()),
            float(np.mean(areas)), float(np.std(areas)), float(np.mean(aspect) if aspect else 0.0),
        ], dtype=np.float32)
        return feat

    def extract_all(self, dataset_scan: MultiDatasetScan, config: dict, save_path: Path, progress_callback):
        raw_normal, raw_anomaly, hc_normal, hc_anomaly = [], [], [], []
        labels, video_ids = [], []
        total = len(dataset_scan.videos)
        for i, video in enumerate(dataset_scan.videos):
            clips = self.loader.iter_clips(
                video.path,
                fps=config.get("fps", 5),
                resize=config.get("resize", 64),
                clip_len=config.get("clip_len", 16),
                max_clips=config.get("max_clips_per_video", 20),
                augment=config.get("augment_normal", True) and video.label == 0,
            )
            for clip in clips:
                raw = self.extract_raw(clip)
                hc = self.extract_handcrafted(clip)
                if video.label == 0:
                    raw_normal.append(raw); hc_normal.append(hc)
                else:
                    raw_anomaly.append(raw); hc_anomaly.append(hc)
                labels.append(video.label)
                video_ids.append(str(video.path))
            progress_callback({"phase": "extract", "video": video.path.name, "label": video.label, "clips": len(clips), "done": i + 1, "total": total, "pct": round((i + 1) * 100 / max(total, 1), 2)})

        np.savez_compressed(
            save_path,
            raw_normal=np.array(raw_normal, dtype=np.float32),
            raw_anomaly=np.array(raw_anomaly, dtype=np.float32),
            hc_normal=np.array(hc_normal, dtype=np.float32),
            hc_anomaly=np.array(hc_anomaly, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
            video_ids=np.array(video_ids),
        )
        return {"total_clips": len(labels), "normal": int(np.sum(np.array(labels) == 0)), "anomaly": int(np.sum(np.array(labels) == 1)), "feature_path": str(save_path), "feature_dim": 20}
