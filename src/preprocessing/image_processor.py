"""
Branch 1 — Image Restoration & Enhancement
==========================================
Provides a configurable preprocessing pipeline:
  1. Resize
  2. Denoising  (Gaussian / Median / Bilateral / Non-Local Means)
  3. Enhancement (CLAHE / Histogram EQ / Gamma Correction)
  4. Optional sharpening
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Low-level helpers ─────────────────────────────────────────────────────

def denoise_gaussian(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Apply Gaussian blur for mild noise removal."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def denoise_median(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Apply Median filter – good for salt-and-pepper noise."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(image, k)


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Bilateral filter – removes noise while preserving edges."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(image: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-Local Means denoising – slower but highest quality."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)


def enhance_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Applied to L channel in LAB color space – better than global HE.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_histogram_eq(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization on V channel (HSV)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    merged = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def enhance_gamma(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Gamma correction. gamma>1 brightens, gamma<1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp masking sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def wiener_deblur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Simple approximation of Wiener deconvolution for motion blur.
    Uses Gaussian kernel as PSF estimate.
    """
    psf = cv2.getGaussianKernel(kernel_size, -1)
    psf = psf @ psf.T
    channels = cv2.split(image)
    deblurred = []
    for ch in channels:
        ch_f = np.fft.fft2(ch.astype(np.float64))
        psf_f = np.fft.fft2(psf, s=ch.shape)
        # Wiener filter with small noise constant K
        K = 0.01
        psf_conj = np.conj(psf_f)
        wiener = psf_conj / (np.abs(psf_f) ** 2 + K)
        restored = np.real(np.fft.ifft2(ch_f * wiener))
        deblurred.append(np.clip(restored, 0, 255).astype(np.uint8))
    return cv2.merge(deblurred)


# ── Pipeline class ────────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Configurable preprocessing pipeline for CCTV frames.

    Example:
        preprocessor = ImagePreprocessor(cfg)
        clean_frame = preprocessor.process(raw_frame)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.resize_wh: Optional[Tuple[int, int]] = tuple(cfg.get("resize", [640, 640]))
        self.denoise_method: str = cfg.get("denoise", {}).get("method", "gaussian")
        self.denoise_kernel: int = cfg.get("denoise", {}).get("kernel_size", 5)
        self.enhance_method: str = cfg.get("enhancement", {}).get("method", "clahe")
        self.clip_limit: float   = cfg.get("enhancement", {}).get("clip_limit", 2.0)
        self.tile_grid           = tuple(cfg.get("enhancement", {}).get("tile_grid", [8, 8]))
        self.do_sharpen: bool    = cfg.get("sharpen", False)
        logger.info(f"ImagePreprocessor ready | denoise={self.denoise_method} | enhance={self.enhance_method}")

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run full preprocessing pipeline on a single BGR image.

        Steps: resize → denoise → enhance → sharpen (optional)
        """
        img = self._resize(image)
        img = self._denoise(img)
        img = self._enhance(img)
        if self.do_sharpen:
            img = sharpen(img)
        return img

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Process a list of images."""
        return [self.process(img) for img in images]

    def compare(self, image: np.ndarray) -> np.ndarray:
        """Return side-by-side before/after comparison image."""
        processed = self.process(image)
        orig_resized = cv2.resize(image, (processed.shape[1], processed.shape[0]))

        # Add labels
        orig_label = orig_resized.copy()
        proc_label = processed.copy()
        cv2.putText(orig_label, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(proc_label, "PROCESSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
        return np.hstack([orig_label, proc_label])

    # ── Private helpers ───────────────────────────────────────────────────

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if self.resize_wh and img.shape[:2][::-1] != tuple(self.resize_wh):
            return cv2.resize(img, self.resize_wh, interpolation=cv2.INTER_LINEAR)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        k = self.denoise_kernel
        if self.denoise_method == "gaussian":
            return denoise_gaussian(img, k)
        if self.denoise_method == "median":
            return denoise_median(img, k)
        if self.denoise_method == "bilateral":
            return denoise_bilateral(img)
        if self.denoise_method == "nlm":
            return denoise_nlm(img)
        return img

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        if self.enhance_method == "clahe":
            return enhance_clahe(img, self.clip_limit, self.tile_grid)
        if self.enhance_method == "histogram_eq":
            return enhance_histogram_eq(img)
        if self.enhance_method == "gamma":
            return enhance_gamma(img)
        return img
