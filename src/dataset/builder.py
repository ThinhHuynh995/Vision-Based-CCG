"""
src/dataset/builder.py — Dataset Builder Pipeline
===================================================

Converts raw CCTV footage + label files into a clean, balanced
train/val/test split ready for training.

Pipeline:
    1. Ingest: scan raw video folder + label JSON files
    2. Extract: pull labeled frames from videos → crop person ROIs
    3. Balance: oversample minority classes to target ratio
    4. Split:   stratified train/val/test split
    5. Verify:  sanity checks + class distribution report

Usage:
    from src.dataset.builder import DatasetBuilder
    builder = DatasetBuilder(cfg)
    stats = builder.build(
        labeled_dir="data/labeled",
        output_dir="data/processed",
    )
"""
from __future__ import annotations
import cv2
import json
import shutil
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {
    0: "Normal",
    1: "Fighting",
    2: "Falling",
    3: "Loitering",
    4: "Crowd Panic",
}


@dataclass
class Sample:
    """Single labeled image sample."""
    path: Path
    class_id: int
    class_name: str
    source_video: str = ""
    frame_idx: int = 0

    @property
    def checksum(self) -> str:
        """MD5 of file for deduplication."""
        with open(self.path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


@dataclass
class DatasetStats:
    """Summary statistics for a built dataset."""
    total_samples: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    train_dist: Dict[str, int] = field(default_factory=dict)
    val_dist: Dict[str, int] = field(default_factory=dict)
    test_dist: Dict[str, int] = field(default_factory=dict)
    duplicate_removed: int = 0
    output_dir: str = ""

    def print(self):
        print("\n" + "═" * 55)
        print("  DATASET BUILD SUMMARY")
        print("═" * 55)
        print(f"  Total samples : {self.total_samples}")
        print(f"  Train         : {self.train_count}")
        print(f"  Val           : {self.val_count}")
        print(f"  Test          : {self.test_count}")
        print(f"  Duplicates rm : {self.duplicate_removed}")
        print()
        print("  Class distribution (train):")
        for cls, n in sorted(self.train_dist.items()):
            bar = "█" * min(n // 2, 30)
            print(f"    {cls:<14} {bar:30s} {n}")
        print(f"\n  Output → {self.output_dir}")
        print("═" * 55 + "\n")


class DatasetBuilder:
    """
    End-to-end dataset builder for CCTV behavior classification.

    Args:
        cfg: project config dict (from config.yaml)
    """

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.seed      = cfg.get("seed", 42)
        random.seed(self.seed)

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        labeled_dir: str = "data/labeled",
        output_dir: str  = "data/processed",
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.10,
        target_per_class: Optional[int] = None,
        remove_duplicates: bool = True,
        min_size: Tuple[int, int] = (32, 32),
    ) -> DatasetStats:
        """
        Run the full build pipeline.

        Args:
            labeled_dir:       root with labeled/<video_stem>/{class}/frames
            output_dir:        root for processed/train|val|test/<class>/
            val_ratio:         fraction for validation set
            test_ratio:        fraction for test set
            target_per_class:  oversample to this count (None = no balancing)
            remove_duplicates: skip identical images (MD5 hash)
            min_size:          (w, h) minimum crop size to include

        Returns:
            DatasetStats object
        """
        logger.info("=== DatasetBuilder: starting pipeline ===")
        labeled_root = Path(labeled_dir)
        out_root     = Path(output_dir)

        # ── Step 1: Collect all samples ───────────────────────────────────
        samples = self._collect_samples(labeled_root, min_size)
        logger.info(f"Collected {len(samples)} raw samples")

        # ── Step 2: Remove duplicates ─────────────────────────────────────
        n_before = len(samples)
        if remove_duplicates:
            samples = self._deduplicate(samples)
        dup_removed = n_before - len(samples)
        logger.info(f"After dedup: {len(samples)} samples ({dup_removed} removed)")

        # ── Step 3: Balance classes ───────────────────────────────────────
        if target_per_class:
            samples = self._balance(samples, target_per_class, labeled_root)
            logger.info(f"After balancing: {len(samples)} samples")

        # ── Step 4: Stratified split ──────────────────────────────────────
        splits = self._stratified_split(samples, val_ratio, test_ratio)

        # ── Step 5: Write to disk ─────────────────────────────────────────
        self._write_split(splits, out_root)

        # ── Step 6: Compute stats ─────────────────────────────────────────
        stats = self._compute_stats(splits, dup_removed, str(out_root))
        stats.print()
        self._write_manifest(splits, out_root)
        return stats

    def extract_frames_from_video(
        self,
        video_path: str,
        label_json: str,
        output_dir: str,
        crop_persons: bool = True,
    ) -> int:
        """
        Extract labeled frames from a video using a label JSON file.

        label_json format: {"frame_idx": label_id, ...}
        Optionally runs HOG person detection to extract tight crops.

        Returns number of frames extracted.
        """
        from src.detection.detector import PersonDetector

        video_path = Path(video_path)
        out_root   = Path(output_dir)
        with open(label_json) as f:
            label_data = {int(k): v for k, v in json.load(f).items()}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return 0

        det = PersonDetector(self.cfg.get("detection", {})) if crop_persons else None
        count = 0

        for frame_idx, label_id in sorted(label_data.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            cls_name = CLASS_NAMES.get(label_id, "Normal")
            save_dir = out_root / cls_name
            save_dir.mkdir(parents=True, exist_ok=True)

            if crop_persons and det:
                dets = det.detect(frame)
                if dets:
                    for i, d in enumerate(dets[:3]):   # max 3 persons per frame
                        x1, y1, x2, y2 = d.bbox
                        crop = frame[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            fname = f"{video_path.stem}_f{frame_idx:06d}_p{i}.jpg"
                            cv2.imwrite(str(save_dir / fname), crop)
                            count += 1
                else:
                    # No detection → save full frame
                    fname = f"{video_path.stem}_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(save_dir / fname), frame)
                    count += 1
            else:
                fname = f"{video_path.stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(save_dir / fname), frame)
                count += 1

        cap.release()
        logger.info(f"Extracted {count} frames from {video_path.name}")
        return count

    # ── Private helpers ───────────────────────────────────────────────────

    def _collect_samples(
        self, root: Path, min_size: Tuple[int, int]
    ) -> List[Sample]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        samples = []
        for cls_id, cls_name in CLASS_NAMES.items():
            cls_dir = root / cls_name
            if not cls_dir.exists():
                # Also scan nested labeled/<video>/ClassName/
                nested = list(root.glob(f"*/{cls_name}"))
                dirs_to_scan = nested
            else:
                dirs_to_scan = [cls_dir]
            for d in dirs_to_scan:
                for p in d.iterdir():
                    if p.suffix.lower() not in exts:
                        continue
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    if w < min_size[0] or h < min_size[1]:
                        continue
                    samples.append(Sample(
                        path=p, class_id=cls_id, class_name=cls_name
                    ))
        return samples

    def _deduplicate(self, samples: List[Sample]) -> List[Sample]:
        seen, unique = set(), []
        for s in samples:
            cs = s.checksum
            if cs not in seen:
                seen.add(cs)
                unique.append(s)
        return unique

    def _balance(
        self, samples: List[Sample], target: int, labeled_root: Path
    ) -> List[Sample]:
        """Oversample minority classes by duplicating (no augmentation here)."""
        by_class: Dict[int, List[Sample]] = defaultdict(list)
        for s in samples:
            by_class[s.class_id].append(s)

        balanced = []
        for cls_id in CLASS_NAMES:
            cls_samples = by_class.get(cls_id, [])
            if not cls_samples:
                logger.warning(f"No samples for class {CLASS_NAMES[cls_id]}")
                continue
            if len(cls_samples) >= target:
                balanced.extend(random.sample(cls_samples, target))
            else:
                # Duplicate with replacement
                extra = random.choices(cls_samples, k=target - len(cls_samples))
                balanced.extend(cls_samples + extra)
        return balanced

    def _stratified_split(
        self,
        samples: List[Sample],
        val_ratio: float,
        test_ratio: float,
    ) -> Dict[str, List[Sample]]:
        by_class: Dict[int, List[Sample]] = defaultdict(list)
        for s in samples:
            by_class[s.class_id].append(s)

        train, val, test = [], [], []
        for cls_id, cls_samples in by_class.items():
            random.shuffle(cls_samples)
            n     = len(cls_samples)
            n_val = max(1, int(n * val_ratio))
            n_tst = max(1, int(n * test_ratio))
            test.extend(cls_samples[:n_tst])
            val.extend( cls_samples[n_tst:n_tst + n_val])
            train.extend(cls_samples[n_tst + n_val:])

        random.shuffle(train)
        return {"train": train, "val": val, "test": test}

    def _write_split(
        self, splits: Dict[str, List[Sample]], out_root: Path
    ):
        for split_name, split_samples in splits.items():
            for s in split_samples:
                dst_dir = out_root / split_name / s.class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / s.path.name
                # Handle name collisions
                if dst_path.exists():
                    stem = s.path.stem
                    suffix = s.path.suffix
                    i = 1
                    while dst_path.exists():
                        dst_path = dst_dir / f"{stem}_{i}{suffix}"
                        i += 1
                shutil.copy2(s.path, dst_path)

    def _compute_stats(
        self,
        splits: Dict[str, List[Sample]],
        dup_removed: int,
        out_dir: str,
    ) -> DatasetStats:
        def dist(lst):
            c = Counter(s.class_name for s in lst)
            return dict(c)

        all_samples = splits["train"] + splits["val"] + splits["test"]
        return DatasetStats(
            total_samples   = len(all_samples),
            train_count     = len(splits["train"]),
            val_count       = len(splits["val"]),
            test_count      = len(splits["test"]),
            class_distribution = dist(all_samples),
            train_dist      = dist(splits["train"]),
            val_dist        = dist(splits["val"]),
            test_dist       = dist(splits["test"]),
            duplicate_removed = dup_removed,
            output_dir      = out_dir,
        )

    def _write_manifest(
        self, splits: Dict[str, List[Sample]], out_root: Path
    ):
        manifest = {
            split: [
                {"path": str(s.path.name), "class": s.class_name, "id": s.class_id}
                for s in lst
            ]
            for split, lst in splits.items()
        }
        with open(out_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written → {out_root / 'manifest.json'}")


# ── CLI shortcut ──────────────────────────────────────────────────────────

def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="Build training dataset from labeled CCTV data")
    parser.add_argument("--labeled",  default="data/labeled",   help="Labeled data root")
    parser.add_argument("--output",   default="data/processed",  help="Output root")
    parser.add_argument("--val",      type=float, default=0.15,  help="Val ratio")
    parser.add_argument("--test",     type=float, default=0.10,  help="Test ratio")
    parser.add_argument("--balance",  type=int,   default=None,  help="Target samples per class")
    parser.add_argument("--config",   default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    builder = DatasetBuilder(cfg)
    builder.build(
        labeled_dir      = args.labeled,
        output_dir       = args.output,
        val_ratio        = args.val,
        test_ratio       = args.test,
        target_per_class = args.balance,
    )


if __name__ == "__main__":
    main()


# ── Static Image Folder Ingestor ──────────────────────────────────────────

class ImageFolderIngestor:
    """
    Ingest a flat folder of camera screenshots that are already
    organized by class name (mirrors ImageNet folder convention):

        raw_images/
            Normal/        ← CCTV screenshots of normal activity
            Fighting/
            Falling/
            Loitering/
            Crowd Panic/

    Copies everything into labeled/<folder_name>/<class>/ so that
    DatasetBuilder.build() can pick it up in the next step.

    Usage:
        ingestor = ImageFolderIngestor()
        ingestor.ingest("raw_images", "data/labeled/static_cam")
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def ingest(
        self,
        src_root: str,
        out_root: str,
        resize: tuple = None,
    ) -> Dict[str, int]:
        """
        Copy + optionally resize images from src_root into out_root.
        Returns dict of {class_name: count}.
        """
        import shutil, cv2 as _cv2
        src = Path(src_root)
        dst = Path(out_root)
        counts = {}
        for cls_name in CLASS_NAMES.values():
            cls_src = src / cls_name
            if not cls_src.exists():
                continue
            cls_dst = dst / cls_name
            cls_dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for p in cls_src.iterdir():
                if p.suffix.lower() not in self.IMG_EXTS:
                    continue
                if resize:
                    img = _cv2.imread(str(p))
                    if img is None:
                        continue
                    img = _cv2.resize(img, resize)
                    _cv2.imwrite(str(cls_dst / p.name), img)
                else:
                    shutil.copy2(p, cls_dst / p.name)
                n += 1
            counts[cls_name] = n
            logger.info(f"Ingested {n} images: {cls_name}")
        return counts
