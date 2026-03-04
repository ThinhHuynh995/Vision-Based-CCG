"""
Utility: Load and access YAML config file.
"""
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class Config:
    """Dot-access wrapper around config dict."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self._data = load_config(config_path)

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._data.get(key)
        if isinstance(val, dict):
            return _DictWrapper(val)
        return val

    def get(self, key, default=None):
        return self._data.get(key, default)


class _DictWrapper:
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._d.get(key)
        if isinstance(val, dict):
            return _DictWrapper(val)
        return val

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)
