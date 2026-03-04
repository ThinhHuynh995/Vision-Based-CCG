"""src/utils/log.py — Logger màu cho console + file."""
from __future__ import annotations
import logging, sys
from pathlib import Path
from datetime import datetime

_registry: dict[str, logging.Logger] = {}


def get(name: str, log_dir: str = "outputs/reports") -> logging.Logger:
    if name in _registry:
        return _registry[name]

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_Color())
    logger.addHandler(ch)

    # File
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh  = logging.FileHandler(
        Path(log_dir) / f"{name.split('.')[-1]}_{ts}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(fh)
    _registry[name] = logger
    return logger


class _Color(logging.Formatter):
    _C = {
        logging.DEBUG:    "\033[37m",
        logging.INFO:     "\033[36m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    _R = "\033[0m"

    def format(self, r: logging.LogRecord) -> str:
        c = self._C.get(r.levelno, self._R)
        r.levelname = f"{c}{r.levelname:<8}{self._R}"
        return logging.Formatter(
            "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        ).format(r)
