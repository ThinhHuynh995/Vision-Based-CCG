"""src/utils/config.py — Config loader với dot-access."""
from __future__ import annotations
import yaml
from pathlib import Path


def load(path: str = "configs/config.yaml") -> "Cfg":
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config không tìm thấy: {path}")
    with open(p, encoding="utf-8") as f:
        return Cfg(yaml.safe_load(f))


class Cfg:
    """Wrapper dict cho phép truy cập cfg.key.subkey."""

    def __init__(self, data: dict):
        self._d = data

    def __getattr__(self, key: str):
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._d.get(key)
        if isinstance(val, dict):
            return Cfg(val)
        return val

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)

    def raw(self) -> dict:
        """Trả về dict gốc (dùng khi cần pass vào hàm nhận dict)."""
        return self._d
