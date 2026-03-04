"""
Branch 4 — Data Augmentation
=============================
  • ImageAugmentor: Albumentations-based augmentation pipeline
  • MixUpAugmentor: MixUp and CutMix for training
  • DatasetBalancer: oversample rare classes
  • generate_augmented_dataset(): CLI-usable batch generator
"""
from __future__ import annotations
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Albumentations pipeline ───────────────────────────────────────────────

class ImageAugmentor:
    """
    Rich augmentation pipeline using Albumentations.

    Falls back to OpenCV transforms when Albumentations is not installed.

    Usage:
        aug = ImageAugmentor(cfg["augmentation"])
        augmented = aug.augment(image)
        grid = aug.show_grid(image, n=8)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pipeline = None
        self._try_build_albumentations(cfg)

    def _try_build_albumentations(self, cfg: dict):
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2  # noqa

            self._pipeline = A.Compose([
                A.HorizontalFlip(p=cfg.get("p_flip", 0.5)),
                A.Rotate(limit=cfg.get("max_rotate_deg", 15), p=cfg.get("p_rotate", 0.3)),
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=cfg.get("p_brightness", 0.4),
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=cfg.get("p_blur", 0.2)),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
                A.RandomShadow(p=0.15),
                A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
                A.Perspective(scale=(0.02, 0.07), p=0.2),
            ])
            logger.info("ImageAugmentor: using Albumentations pipeline")
        except ImportError:
            logger.warning("Albumentations not installed – using OpenCV fallback")
            self._pipeline = None

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to a BGR image."""
        if self._pipeline is not None:
            result = self._pipeline(image=image)
            return result["image"]
        return self._opencv_augment(image)

    def _opencv_augment(self, image: np.ndarray) -> np.ndarray:
        """Pure OpenCV augmentation fallback."""
        img = image.copy()

        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Brightness / contrast
        if random.random() < 0.4:
            alpha = random.uniform(0.7, 1.3)  # contrast
            beta  = random.randint(-30, 30)   # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Gaussian blur
        if random.random() < 0.2:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            h, w  = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        return img

    def augment_batch(self, images: List[np.ndarray], n_per_image: int = 3) -> List[np.ndarray]:
        """Generate n augmented versions per input image."""
        results = []
        for img in images:
            for _ in range(n_per_image):
                results.append(self.augment(img))
        return results

    def show_grid(self, image: np.ndarray, n: int = 8) -> np.ndarray:
        """Return a grid of n augmented versions for visual inspection."""
        cols = 4
        rows = (n + cols - 1) // cols
        h, w = image.shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i in range(n):
            aug = self.augment(image)
            r, c = divmod(i, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(aug, (w, h))
        return grid


# ── MixUp & CutMix ────────────────────────────────────────────────────────

class MixUpAugmentor:
    """
    Implements MixUp and CutMix augmentation strategies.

    Usage (in training loop):
        mixup = MixUpAugmentor(alpha=0.4)
        mixed_img, lam, label_a, label_b = mixup.mixup(img_a, img_b, lbl_a, lbl_b)
        # loss = lam * criterion(out, label_a) + (1-lam) * criterion(out, label_b)
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def mixup(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label_a: int,
        label_b: int,
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Blend two images linearly.

        Returns:
            (mixed_image, lambda, label_a, label_b)
        """
        lam = np.random.beta(self.alpha, self.alpha)
        img_b_resized = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        mixed = (lam * img_a.astype(np.float32) +
                 (1 - lam) * img_b_resized.astype(np.float32)).astype(np.uint8)
        return mixed, lam, label_a, label_b

    def cutmix(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label_a: int,
        label_b: int,
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Paste a random rectangular patch from img_b onto img_a.

        Returns:
            (mixed_image, lambda, label_a, label_b)
        """
        h, w = img_a.shape[:2]
        img_b_r = cv2.resize(img_b, (w, h))
        mixed = img_a.copy()

        # Sample patch size from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))

        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)

        mixed[y1:y2, x1:x2] = img_b_r[y1:y2, x1:x2]
        # Recompute lambda from actual patch area
        lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
        return mixed, lam, label_a, label_b


# ── Dataset Generator ─────────────────────────────────────────────────────

def generate_augmented_dataset(
    src_dir: str,
    dst_dir: str,
    n_per_image: int = 5,
    cfg: Optional[dict] = None,
) -> None:
    """
    Batch-augment all images from src_dir and save to dst_dir,
    preserving class subdirectory structure.

    Args:
        src_dir:      source dataset root (with class subfolders)
        dst_dir:      destination root
        n_per_image:  number of augmented copies per image
        cfg:          augmentation config dict
    """
    augmentor = ImageAugmentor(cfg or {})
    src = Path(src_dir)
    dst = Path(dst_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images_found = list(src.rglob("*"))
    images_found = [p for p in images_found if p.suffix.lower() in exts]

    logger.info(f"Augmenting {len(images_found)} images → {dst}")

    for img_path in images_found:
        rel = img_path.relative_to(src)
        out_dir = dst / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Copy original
        cv2.imwrite(str(out_dir / img_path.name), img)

        # Write augmented versions
        for i in range(n_per_image):
            aug = augmentor.augment(img)
            stem = img_path.stem
            out_path = out_dir / f"{stem}_aug{i:02d}{img_path.suffix}"
            cv2.imwrite(str(out_path), aug)

    logger.info(f"Done. Augmented dataset saved to {dst}")
