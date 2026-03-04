"""
src/augmentation/augmentor.py — Branch 4: Data Augmentation
============================================================
  • Augmentor     – Pipeline Albumentations (fallback: OpenCV)
  • MixUp/CutMix  – Advanced blending strategies
  • augment_dir() – Batch CLI: nhân bội dataset
"""
from __future__ import annotations
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Optional, Tuple
import src.utils.log as _log

log = _log.get(__name__)


class Augmentor:
    """
    Augmentation pipeline.
    Dùng Albumentations nếu cài, ngược lại dùng OpenCV thuần.

    Sử dụng:
        aug = Augmentor(cfg.augmentation)
        out = aug(image_bgr)
        grid = aug.grid(image_bgr, n=8)
    """

    def __init__(self, cfg=None):
        self._pipeline = None
        cfg = cfg or {}
        self._p_flip   = float(getattr(cfg, "p_flip",   0) or cfg.get("p_flip",   0.5) if hasattr(cfg,"get") else 0.5)
        self._p_rot    = float(getattr(cfg, "p_rotate", 0) or 0.35)
        self._max_rot  = float(getattr(cfg, "max_rotate_deg", 0) or 15)
        self._p_bri    = float(getattr(cfg, "p_brightness", 0) or 0.4)
        self._p_blur   = float(getattr(cfg, "p_blur",   0) or 0.2)
        self._try_albumentations()

    def _try_albumentations(self):
        try:
            import albumentations as A
            self._pipeline = A.Compose([
                A.HorizontalFlip(p=self._p_flip),
                A.Rotate(limit=int(self._max_rot), p=self._p_rot),
                A.RandomBrightnessContrast(brightness_limit=0.25,
                                           contrast_limit=0.25, p=self._p_bri),
                A.GaussianBlur(blur_limit=(3,7), p=self._p_blur),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
                A.RandomShadow(p=0.12),
                A.CoarseDropout(max_holes=4, max_height=28, max_width=28, p=0.18),
                A.Perspective(scale=(0.02, 0.07), p=0.18),
            ])
            log.info("Augmentor: Albumentations")
        except ImportError:
            log.warning("Albumentations chưa cài → dùng OpenCV fallback")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._pipeline is not None:
            return self._pipeline(image=img)["image"]
        return self._cv(img)

    def _cv(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        if random.random() < self._p_flip:
            out = cv2.flip(out, 1)
        if random.random() < self._p_bri:
            a = random.uniform(0.7, 1.3)
            b = random.randint(-30, 30)
            out = cv2.convertScaleAbs(out, alpha=a, beta=b)
        if random.random() < self._p_blur:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        if random.random() < self._p_rot:
            angle = random.uniform(-self._max_rot, self._max_rot)
            h, w  = out.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h))
        return out

    def grid(self, img: np.ndarray, n: int = 8) -> np.ndarray:
        """Lưới n ảnh augmented để kiểm tra trực quan."""
        cols = 4
        rows = (n + cols - 1) // cols
        h, w = img.shape[:2]
        grid = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
        for i in range(n):
            r, c = divmod(i, cols)
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = cv2.resize(self(img), (w, h))
        return grid


class MixUp:
    """MixUp và CutMix cho training loop."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def mixup(self, a: np.ndarray, b: np.ndarray,
              la: int, lb: int) -> Tuple[np.ndarray, float, int, int]:
        lam = np.random.beta(self.alpha, self.alpha)
        b_r = cv2.resize(b, (a.shape[1], a.shape[0]))
        mixed = (lam*a.astype(np.float32) + (1-lam)*b_r.astype(np.float32)).astype(np.uint8)
        return mixed, lam, la, lb

    def cutmix(self, a: np.ndarray, b: np.ndarray,
               la: int, lb: int) -> Tuple[np.ndarray, float, int, int]:
        h, w = a.shape[:2]
        b_r  = cv2.resize(b, (w, h))
        lam  = np.random.beta(self.alpha, self.alpha)
        cw, ch = int(w * np.sqrt(1-lam)), int(h * np.sqrt(1-lam))
        cx, cy = random.randint(0, w), random.randint(0, h)
        x1,x2 = max(cx-cw//2,0), min(cx+cw//2,w)
        y1,y2 = max(cy-ch//2,0), min(cy+ch//2,h)
        mixed = a.copy()
        mixed[y1:y2, x1:x2] = b_r[y1:y2, x1:x2]
        lam = 1 - (x2-x1)*(y2-y1)/(w*h)
        return mixed, lam, la, lb


def augment_dir(src: str, dst: str, n: int = 5, cfg=None) -> int:
    """
    Nhân bội toàn bộ ảnh trong src → dst, giữ cấu trúc thư mục.
    Trả về tổng số ảnh đã tạo.

    Sử dụng:
        augment_dir("data/processed/train", "data/augmented/train", n=5)
    """
    aug   = Augmentor(cfg)
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in Path(src).rglob("*") if p.suffix.lower() in exts]
    total = 0
    log.info(f"Augmenting {len(files)} ảnh → {dst}")

    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rel     = img_path.relative_to(src)
        out_dir = Path(dst) / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy gốc
        cv2.imwrite(str(out_dir / img_path.name), img)
        total += 1
        # Augmented copies
        for i in range(n):
            aug_img = aug(img)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_aug{i:02d}{img_path.suffix}"), aug_img)
            total += 1

    log.info(f"Done. {total} ảnh → {dst}")
    return total
