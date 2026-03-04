"""
src/preprocessing/processor.py — Branch 1: Image Restoration & Enhancement
============================================================================
Pipeline: resize → denoise → enhance → sharpen (optional)
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional
import src.utils.log as _log

log = _log.get(__name__)

# ── Hàm đơn lẻ ──────────────────────────────────────────────────────────────

def denoise(img: np.ndarray, method: str = "gaussian", k: int = 5) -> np.ndarray:
    k = k if k % 2 == 1 else k + 1
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "median":
        return cv2.medianBlur(img, k)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    if method == "nlm":
        return (cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                if img.ndim == 3 else cv2.fastNlMeansDenoising(img, None, 10, 7, 21))
    return img


def enhance(
    img: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid: Tuple[int,int] = (8,8),
) -> np.ndarray:
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if method == "histogram_eq":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return cv2.cvtColor(cv2.merge([h, s, cv2.equalizeHist(v)]), cv2.COLOR_HSV2BGR)
    if method == "gamma":
        table = np.array([(i/255.0)**0.7 * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)
    return img


def sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def wiener_deblur(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Ước lượng Wiener qua FFT – hữu ích khi camera bị rung."""
    psf = cv2.getGaussianKernel(k, -1)
    psf = psf @ psf.T
    channels = cv2.split(img)
    out = []
    for ch in channels:
        F  = np.fft.fft2(ch.astype(np.float64))
        Fp = np.fft.fft2(psf, s=ch.shape)
        W  = np.conj(Fp) / (np.abs(Fp)**2 + 0.01)
        out.append(np.clip(np.real(np.fft.ifft2(F * W)), 0, 255).astype(np.uint8))
    return cv2.merge(out)


# ── Pipeline class ──────────────────────────────────────────────────────────

class Preprocessor:
    """
    Preprocessing pipeline cấu hình hoàn toàn qua config.yaml.

    Sử dụng:
        pp = Preprocessor(cfg.preprocessing)
        clean = pp(frame)
        comparison = pp.compare(frame)   # ảnh before|after
    """

    def __init__(self, cfg):
        self._wh     = tuple(cfg.resize) if cfg.resize else (640, 640)
        self._dn_m   = cfg.denoise.method      if cfg.denoise    else "gaussian"
        self._dn_k   = cfg.denoise.kernel_size if cfg.denoise    else 5
        self._en_m   = cfg.enhancement.method     if cfg.enhancement else "clahe"
        self._clip   = cfg.enhancement.clip_limit if cfg.enhancement else 2.0
        self._tile   = tuple(cfg.enhancement.tile_grid) if cfg.enhancement else (8,8)
        self._sharp  = bool(cfg.sharpen) if cfg.sharpen is not None else False
        log.info(f"Preprocessor | resize={self._wh} denoise={self._dn_m} enhance={self._en_m}")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Chạy toàn bộ pipeline trên một ảnh BGR."""
        if img is None or img.size == 0:
            return img
        if img.shape[:2][::-1] != self._wh:
            img = cv2.resize(img, self._wh, interpolation=cv2.INTER_LINEAR)
        img = denoise(img, self._dn_m, self._dn_k)
        img = enhance(img, self._en_m, self._clip, self._tile)
        if self._sharp:
            img = sharpen(img)
        return img

    def compare(self, img: np.ndarray) -> np.ndarray:
        """Trả về ảnh before | after để kiểm tra trực quan."""
        processed = self(img)
        orig = cv2.resize(img, (processed.shape[1], processed.shape[0]))
        for txt, frame in [("ORIGINAL", orig), ("PROCESSED", processed)]:
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 255, 255), 2)
        return np.hstack([orig, processed])
