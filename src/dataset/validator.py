"""
src/dataset/validator.py — Dataset Validation & Quality Checks
===============================================================

Runs before training to catch common dataset problems:
  - Corrupt / unreadable images
  - Extreme class imbalance
  - Blurry or too-dark frames
  - Duplicate images across train/val/test (data leakage)
  - Minimum sample count per class

Usage:
    from src.dataset.validator import DatasetValidator
    report = DatasetValidator.run("data/processed")
    report.print()
"""
from __future__ import annotations
import cv2
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "Normal", 1: "Fighting", 2: "Falling", 3: "Loitering", 4: "Crowd Panic"}
MIN_SAMPLES_PER_CLASS = 30


@dataclass
class ValidationReport:
    """Results from a full dataset validation pass."""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def warn(self, msg: str):
        logger.warning(msg)
        self.warnings.append(msg)

    def error(self, msg: str):
        logger.error(msg)
        self.errors.append(msg)
        self.passed = False

    def print(self):
        print("\n" + "═" * 55)
        print(f"  DATASET VALIDATION  {'✅ PASSED' if self.passed else '❌ FAILED'}")
        print("═" * 55)
        for k, v in self.stats.items():
            print(f"  {k:<30}: {v}")
        if self.warnings:
            print(f"\n  ⚠  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"     - {w}")
        if self.errors:
            print(f"\n  ✗  Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"     - {e}")
        print("═" * 55 + "\n")


class DatasetValidator:
    """Validates a processed dataset directory."""

    SPLITS = ["train", "val", "test"]

    @classmethod
    def run(
        cls,
        dataset_root: str,
        min_samples: int = MIN_SAMPLES_PER_CLASS,
        blur_threshold: float = 80.0,
        darkness_threshold: float = 30.0,
        check_leakage: bool = True,
    ) -> ValidationReport:
        """
        Full validation pass.

        Args:
            dataset_root:        path to processed dataset root
            min_samples:         minimum images per class per split
            blur_threshold:      Laplacian variance below this → blurry flag
            darkness_threshold:  mean pixel value below this → too dark
            check_leakage:       detect duplicate images across splits

        Returns:
            ValidationReport
        """
        root   = Path(dataset_root)
        report = ValidationReport()
        logger.info(f"Validating dataset at {root}")

        split_checksums: Dict[str, Dict[str, str]] = {}   # split → {path: hash}

        total_images = 0
        total_corrupt = 0
        total_blurry = 0
        total_dark = 0
        class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for split in cls.SPLITS:
            split_dir = root / split
            if not split_dir.exists():
                if split == "train":
                    report.error(f"Missing required split directory: {split_dir}")
                else:
                    report.warn(f"Missing split directory: {split_dir}")
                continue

            checksums = {}
            for cls_name in CLASS_NAMES.values():
                cls_dir = split_dir / cls_name
                if not cls_dir.exists():
                    report.warn(f"Missing class dir: {split}/{cls_name}")
                    continue

                images = [p for p in cls_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
                class_counts[split][cls_name] = len(images)
                total_images += len(images)

                for img_path in images:
                    # Corrupt check
                    img = cv2.imread(str(img_path))
                    if img is None:
                        report.warn(f"Corrupt image: {img_path.relative_to(root)}")
                        total_corrupt += 1
                        continue

                    # Checksum for leakage detection
                    with open(img_path, "rb") as f:
                        cs = hashlib.md5(f.read()).hexdigest()
                    checksums[str(img_path)] = cs

                    # Blur check (Laplacian variance)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < blur_threshold:
                        total_blurry += 1

                    # Darkness check
                    mean_val = gray.mean()
                    if mean_val < darkness_threshold:
                        total_dark += 1

            split_checksums[split] = checksums

        # ── Class balance check ───────────────────────────────────────────
        if "train" in class_counts:
            train_counts = class_counts["train"]
            if train_counts:
                max_count = max(train_counts.values())
                for cls_name, n in train_counts.items():
                    if n < min_samples:
                        report.error(
                            f"Too few training samples: {cls_name} has {n} "
                            f"(need ≥ {min_samples})"
                        )
                    if max_count > 0 and n / max_count < 0.2:
                        report.warn(
                            f"Severe class imbalance: {cls_name} has {n} vs "
                            f"max {max_count} ({n/max_count*100:.0f}%)"
                        )

        # ── Cross-split leakage check ─────────────────────────────────────
        if check_leakage and len(split_checksums) >= 2:
            splits_list = list(split_checksums.items())
            for i in range(len(splits_list)):
                for j in range(i + 1, len(splits_list)):
                    s1_name, s1_hashes = splits_list[i]
                    s2_name, s2_hashes = splits_list[j]
                    common = set(s1_hashes.values()) & set(s2_hashes.values())
                    if common:
                        report.warn(
                            f"Data leakage: {len(common)} identical images "
                            f"found in both {s1_name} and {s2_name}"
                        )

        # ── Blurry / dark warnings ────────────────────────────────────────
        if total_images > 0:
            blur_pct = total_blurry / total_images * 100
            dark_pct = total_dark / total_images * 100
            if blur_pct > 20:
                report.warn(f"{blur_pct:.1f}% of images appear blurry (Laplacian < {blur_threshold})")
            if dark_pct > 15:
                report.warn(f"{dark_pct:.1f}% of images appear too dark (mean < {darkness_threshold})")

        # ── Stats summary ─────────────────────────────────────────────────
        report.stats = {
            "Total images":       total_images,
            "Corrupt":            total_corrupt,
            "Blurry (flagged)":   total_blurry,
            "Too dark (flagged)": total_dark,
            "Train classes":      dict(class_counts.get("train", {})),
            "Val classes":        dict(class_counts.get("val", {})),
            "Test classes":       dict(class_counts.get("test", {})),
        }

        return report

    @classmethod
    def class_distribution_chart(cls, dataset_root: str) -> None:
        """Print ASCII bar chart of class distribution per split."""
        root = Path(dataset_root)
        print("\nClass Distribution Chart")
        print("─" * 55)
        for split in cls.SPLITS:
            split_dir = root / split
            if not split_dir.exists():
                continue
            print(f"\n  {split.upper()}")
            for cls_name in CLASS_NAMES.values():
                cls_dir = split_dir / cls_name
                n = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
                bar = "█" * min(n // 3, 35)
                print(f"    {cls_name:<14} {bar:<35} {n:4d}")
        print()
