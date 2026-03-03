"""Training script for behavior classification from annotated video segments.

Features:
- Multi-class supervised training with labels:
  normal_flow, abnormal_gathering, pushing_shoving, fighting, vandalism
- Auto-create annotation file from template if missing
- Auto-bootstrap segments (normal_flow) if annotation has no valid segments
- Cache processed frames and extracted feature vectors in /data for fast re-training
- GPU-aware training (CUDA + mixed precision)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services.behavior_labels import BEHAVIOR_LABELS, BEHAVIOR_TO_INDEX
from app.services.behavior_model import BehaviorClassifier
from app.services.pipeline import FrameFeatures, extract_features, process_video_with_processed_frames

DEFAULT_TRAIN_DIRS = [
    r"D:\NWPU-Videos\videos\NWPUCampusDataset\videos\Train",
    r"E:\1. ThS\shanghaitech\shanghaitech\training\videos",
    r"E:\1. ThS\Avenue Dataset\training_videos",
    r"E:\1. ThS\Normal_Videos",

]
DEFAULT_ANNOTATION_PATH = Path("behavior_annotations.json")
DEFAULT_TEMPLATE_PATH = Path("behavior_annotations.example.json")
DEFAULT_CACHE_DIR = Path("processed_behavior_frames")
DEFAULT_CHECKPOINT_PATH = Path("dcad_behavior_model.pt")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def _resolve_video_path(video_ref: str, train_dirs: list[Path]) -> Path | None:
    candidate = Path(video_ref)
    if candidate.exists():
        return candidate

    for root in train_dirs:
        full = root / video_ref
        if full.exists():
            return full

    basename = Path(video_ref).name
    for root in train_dirs:
        if not root.exists():
            continue
        for path in root.rglob(basename):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
    return None


def _ensure_template_exists(template_path: Path) -> None:
    if template_path.exists():
        return

    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_payload = {
        "videos": [
            {
                "video": "D001_01.avi",
                "segments": [
                    {"start_frame": 0, "end_frame": 300, "label": "normal_flow"},
                    {"start_frame": 301, "end_frame": 520, "label": "abnormal_gathering"},
                ],
            }
        ]
    }
    template_path.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Created annotation template: {template_path}")


def _ensure_annotation_exists(annotation_path: Path, template_path: Path) -> None:
    if annotation_path.exists():
        print(f"[INFO] Using annotation file: {annotation_path}")
        return

    _ensure_template_exists(template_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, annotation_path)
    print(f"[INFO] Annotation file not found. Created from template: {annotation_path}")
    print("[INFO] Review and update labels/segments for better training quality.")


def _find_videos(train_dirs: list[Path], max_items: int | None = None) -> list[Path]:
    videos: list[Path] = []
    for root in train_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
                if max_items is not None and len(videos) >= max_items:
                    return videos
    return videos


def _bootstrap_segments_from_videos(
    annotation_path: Path,
    train_dirs: list[Path],
    max_bootstrap_videos: int = 20,
    bootstrap_end_frame: int = 300,
) -> int:
    max_items = None if max_bootstrap_videos <= 0 else max_bootstrap_videos
    videos = _find_videos(train_dirs, max_items=max_items)
    if not videos:
        return 0

    payload = {
        "videos": [
            {
                "video": str(video),
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": bootstrap_end_frame,
                        "label": "normal_flow",
                    }
                ],
            }
            for video in videos
        ]
    }
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(videos)


def _collect_samples(entries: list[dict[str, object]], train_dirs: list[Path]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []

    for item in entries:
        video_ref = str(item.get("video", "")).strip()
        if not video_ref:
            continue

        video_path = _resolve_video_path(video_ref, train_dirs)
        if video_path is None:
            print(f"[WARN] Video not found for annotation: {video_ref}")
            continue

        segments = item.get("segments", [])
        if not isinstance(segments, list):
            continue

        for seg in segments:
            label = str(seg.get("label", "")).strip()
            if label not in BEHAVIOR_TO_INDEX:
                print(f"[WARN] Skip unknown label '{label}' in {video_ref}")
                continue

            start_frame = int(seg.get("start_frame", 0))
            end_raw = seg.get("end_frame")
            end_frame = int(end_raw) if end_raw is not None else None
            if end_frame is not None and end_frame <= start_frame:
                continue

            samples.append(
                {
                    "video_path": video_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "label": label,
                }
            )

    return samples


def load_annotations(
    annotation_path: Path,
    train_dirs: list[Path],
    template_path: Path,
    bootstrap_if_empty: bool,
    bootstrap_max_videos: int,
    bootstrap_end_frame: int,
    force_bootstrap_all: bool,
) -> list[dict[str, object]]:
    _ensure_annotation_exists(annotation_path, template_path)
    if force_bootstrap_all:
        generated = _bootstrap_segments_from_videos(
            annotation_path=annotation_path,
            train_dirs=train_dirs,
            max_bootstrap_videos=bootstrap_max_videos,
            bootstrap_end_frame=bootstrap_end_frame,
        )
        if generated == 0:
            raise RuntimeError("No videos found in train directories for forced bootstrap.")
        print(f"[INFO] Force-bootstrapped {generated} videos into annotation file.")

    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        entries = payload.get("videos", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Unsupported annotation format. Expect dict with 'videos' or list.")

    samples = _collect_samples(entries, train_dirs)
    if samples:
        return samples

    if not bootstrap_if_empty:
        raise RuntimeError("No valid labeled segments found in annotation file.")

    print("[WARN] No valid labeled segments found. Bootstrapping normal_flow segments...")
    generated = _bootstrap_segments_from_videos(
        annotation_path=annotation_path,
        train_dirs=train_dirs,
        max_bootstrap_videos=bootstrap_max_videos,
        bootstrap_end_frame=bootstrap_end_frame,
    )
    if generated == 0:
        raise RuntimeError("No videos found in train directories for bootstrap.")

    print(f"[INFO] Generated {generated} bootstrap videos with label 'normal_flow'.")
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    entries = payload.get("videos", []) if isinstance(payload, dict) else payload
    samples = _collect_samples(entries, train_dirs)
    if not samples:
        raise RuntimeError("Bootstrap succeeded but no valid segments were collected.")
    return samples


def _feature_vector_from_features(features: list[FrameFeatures]) -> np.ndarray | None:
    if not features:
        return None

    mean_intensity = np.array([f.mean_intensity for f in features], dtype=np.float32)
    motion = np.array([f.motion_score for f in features], dtype=np.float32)
    density_hint = float(np.clip(motion.mean() / 6.0, 0.0, 1.0))

    return np.array(
        [
            float(mean_intensity.mean()),
            float(mean_intensity.std()),
            float(motion.mean()),
            float(motion.std()),
            density_hint,
            float(len(features)),
        ],
        dtype=np.float32,
    )


def _cache_key(
    video_path: Path,
    label: str,
    start_frame: int,
    end_frame: int | None,
    max_frames: int,
    frame_stride: int,
    resize_width: int,
) -> str:
    try:
        stat = video_path.stat()
        version = f"{stat.st_size}:{stat.st_mtime_ns}"
    except FileNotFoundError:
        version = "missing"

    raw = (
        f"{video_path.resolve()}|{version}|{label}|{start_frame}|{end_frame}|"
        f"{max_frames}|{frame_stride}|{resize_width}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]


def _save_processed_frames(cache_frames_dir: Path, processed_frames: list[np.ndarray]) -> None:
    cache_frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(processed_frames):
        frame_path = cache_frames_dir / f"frame_{idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)


def _compute_features_from_cached_frames(frame_paths: list[Path]) -> list[FrameFeatures]:
    if len(frame_paths) < 2:
        return []

    prev = cv2.imread(str(frame_paths[0]))
    if prev is None:
        return []

    features: list[FrameFeatures] = []
    for frame_path in frame_paths[1:]:
        cur = cv2.imread(str(frame_path))
        if cur is None:
            continue
        features.append(extract_features(prev, cur))
        prev = cur

    return features


def _extract_or_load_cached_feature(
    sample: dict[str, object],
    max_frames: int,
    frame_stride: int,
    resize_width: int,
    cache_dir: Path | None,
) -> np.ndarray | None:
    video_path = Path(sample["video_path"])
    label = str(sample["label"])
    start_frame = int(sample["start_frame"])
    end_frame = sample["end_frame"]

    if cache_dir is None:
        features, _ = process_video_with_processed_frames(
            video_path=video_path,
            max_frames=max_frames,
            frame_stride=frame_stride,
            resize_width=resize_width,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return _feature_vector_from_features(features)

    key = _cache_key(
        video_path=video_path,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
        max_frames=max_frames,
        frame_stride=frame_stride,
        resize_width=resize_width,
    )

    segment_dir = cache_dir / label / key
    frames_dir = segment_dir / "frames"
    feature_path = segment_dir / "feature.npy"
    meta_path = segment_dir / "meta.json"

    if feature_path.exists():
        return np.load(feature_path).astype(np.float32)

    cached_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if cached_frames:
        features = _compute_features_from_cached_frames(cached_frames)
        vector = _feature_vector_from_features(features)
        if vector is not None:
            segment_dir.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, vector)
            if not meta_path.exists():
                meta = {
                    "video_path": str(video_path),
                    "label": label,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "cache_key": key,
                    "cached_frame_count": len(cached_frames),
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return vector

    features, processed_frames = process_video_with_processed_frames(
        video_path=video_path,
        max_frames=max_frames,
        frame_stride=frame_stride,
        resize_width=resize_width,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    vector = _feature_vector_from_features(features)
    if vector is None:
        return None

    segment_dir.mkdir(parents=True, exist_ok=True)
    _save_processed_frames(frames_dir, processed_frames)
    np.save(feature_path, vector)

    meta = {
        "video_path": str(video_path),
        "label": label,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "cache_key": key,
        "cached_frame_count": len(processed_frames),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return vector


def _extract_sample(
    sample: dict[str, object],
    max_frames: int,
    frame_stride: int,
    resize_width: int,
    cache_dir: Path | None,
) -> tuple[np.ndarray, int] | None:
    try:
        vector = _extract_or_load_cached_feature(
            sample=sample,
            max_frames=max_frames,
            frame_stride=frame_stride,
            resize_width=resize_width,
            cache_dir=cache_dir,
        )
    except Exception:
        return None

    if vector is None:
        return None

    label_idx = BEHAVIOR_TO_INDEX[str(sample["label"])]
    return vector, label_idx


def build_dataset(
    samples: list[dict[str, object]],
    max_frames: int,
    frame_stride: int,
    resize_width: int,
    extraction_workers: int,
    cache_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[np.ndarray] = []
    labels: list[int] = []

    t0 = time.perf_counter()

    if extraction_workers <= 1:
        for idx, sample in enumerate(samples, start=1):
            result = _extract_sample(sample, max_frames, frame_stride, resize_width, cache_dir)
            if result is not None:
                feat, cls_idx = result
                rows.append(feat)
                labels.append(cls_idx)
            if idx % 25 == 0:
                print(f"Processed {idx}/{len(samples)} segments...")
    else:
        with ThreadPoolExecutor(max_workers=extraction_workers) as pool:
            futures = [
                pool.submit(_extract_sample, sample, max_frames, frame_stride, resize_width, cache_dir)
                for sample in samples
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    feat, cls_idx = result
                    rows.append(feat)
                    labels.append(cls_idx)
                if idx % 25 == 0:
                    print(f"Processed {idx}/{len(samples)} segments...")

    elapsed = time.perf_counter() - t0
    print(f"Feature extraction done in {elapsed:.2f}s. Usable segments: {len(rows)}")

    if not rows:
        raise RuntimeError("No segment features extracted.")

    x = torch.from_numpy(np.stack(rows).astype(np.float32))
    y = torch.from_numpy(np.array(labels, dtype=np.int64))
    return x, y


def _class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y, minlength=len(BEHAVIOR_LABELS)).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return weights / weights.mean()


def _label_distribution(y: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(y, minlength=len(BEHAVIOR_LABELS)).tolist()
    return {label: int(counts[idx]) for idx, label in enumerate(BEHAVIOR_LABELS)}


def run_training(
    train_dirs: list[Path],
    annotation_path: Path,
    annotation_template: Path,
    epochs: int,
    batch_size: int,
    max_frames: int,
    frame_stride: int,
    resize_width: int,
    extraction_workers: int,
    cache_dir: Path,
    use_cache: bool,
    bootstrap_if_empty: bool,
    bootstrap_max_videos: int,
    bootstrap_end_frame: int,
    force_bootstrap_all: bool,
    checkpoint_path: Path,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = load_annotations(
        annotation_path=annotation_path,
        train_dirs=train_dirs,
        template_path=annotation_template,
        bootstrap_if_empty=bootstrap_if_empty,
        bootstrap_max_videos=bootstrap_max_videos,
        bootstrap_end_frame=bootstrap_end_frame,
        force_bootstrap_all=force_bootstrap_all,
    )
    print(f"Loaded {len(samples)} labeled segments from {annotation_path}")

    cache_target = cache_dir if use_cache else None
    if cache_target is not None:
        cache_target.mkdir(parents=True, exist_ok=True)
        print(f"Cache dir: {cache_target.resolve()}")

    x, y = build_dataset(
        samples=samples,
        max_frames=max_frames,
        frame_stride=frame_stride,
        resize_width=resize_width,
        extraction_workers=max(1, extraction_workers),
        cache_dir=cache_target,
    )

    print(f"Label distribution: {_label_distribution(y)}")

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = BehaviorClassifier(input_dim=x.shape[1], num_classes=len(BEHAVIOR_LABELS)).to(device)
    weights = _class_weights(y).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0 = time.perf_counter()
    print(f"Start training with {len(dataset)} samples...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0

        for feat, label in loader:
            feat = feat.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(feat)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            seen += label.numel()

        avg_loss = total_loss / max(1, len(loader))
        acc = correct / max(1, seen)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"Training time: {elapsed:.2f}s")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "labels": BEHAVIOR_LABELS,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_frames": max_frames,
                "frame_stride": frame_stride,
                "resize_width": resize_width,
                "extraction_workers": extraction_workers,
                "annotation_path": str(annotation_path),
                "train_dirs": [str(p) for p in train_dirs],
                "cache_dir": str(cache_dir),
                "use_cache": use_cache,
                "bootstrap_if_empty": bootstrap_if_empty,
                "bootstrap_max_videos": bootstrap_max_videos,
                "bootstrap_end_frame": bootstrap_end_frame,
                "force_bootstrap_all": force_bootstrap_all,
                "seed": seed,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior classifier from annotated video segments.")
    parser.add_argument("--train-dirs", nargs="+", default=DEFAULT_TRAIN_DIRS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--annotation-template", type=Path, default=DEFAULT_TEMPLATE_PATH)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--resize-width", type=int, default=480)
    parser.add_argument("--extraction-workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))

    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", help="Disable reading/writing cache.")

    parser.add_argument(
        "--no-bootstrap-if-empty",
        action="store_true",
        help="If annotation has no valid segments, fail instead of auto-bootstrapping normal_flow segments.",
    )
    parser.add_argument(
        "--bootstrap-max-videos",
        type=int,
        default=20,
        help="Max videos used when bootstrapping annotation. Use 0 for all videos.",
    )
    parser.add_argument(
        "--bootstrap-end-frame",
        type=int,
        default=300,
        help="End frame used for auto-generated normal_flow segment.",
    )
    parser.add_argument(
        "--bootstrap-all-videos",
        action="store_true",
        help="Ignore existing annotation and regenerate from videos in train dirs.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        train_dirs=[Path(p) for p in args.train_dirs],
        annotation_path=args.annotations,
        annotation_template=args.annotation_template,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        resize_width=args.resize_width,
        extraction_workers=args.extraction_workers,
        cache_dir=args.cache_dir,
        use_cache=(not args.no_cache),
        bootstrap_if_empty=(not args.no_bootstrap_if_empty),
        bootstrap_max_videos=args.bootstrap_max_videos,
        bootstrap_end_frame=args.bootstrap_end_frame,
        force_bootstrap_all=args.bootstrap_all_videos,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
