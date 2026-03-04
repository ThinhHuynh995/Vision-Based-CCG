"""
Utility: Centralized logger with color output.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "outputs/reports") -> logging.Logger:
    """
    Create and return a logger that writes to both console and file.

    Args:
        name: Logger name (usually __name__ of calling module)
        log_dir: Directory to write log files

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ──────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # ── File handler ─────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name.replace('.', '_')}_{ts}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[37m",   # grey
        logging.INFO:     "\033[36m",   # cyan
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        fmt = "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)
