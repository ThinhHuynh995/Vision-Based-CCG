"""
src/io/video_source.py — Unified Video Source Abstraction
==========================================================

Provides a single interface for reading frames from:
  • MP4 / AVI files
  • RTSP streams (IP cameras)
  • Webcam (index or device path)

Also includes:
  • FrameBuffer: thread-safe frame queue for smooth reading
  • VideoWriter: annotated output writer with metadata overlay

Usage:
    source = VideoSource.from_config(cfg)
    with source:
        for frame in source:
            process(frame)
"""
from __future__ import annotations
import cv2
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Iterator, Union
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameMeta:
    """Metadata attached to each captured frame."""
    frame_idx: int
    timestamp: float        # seconds from stream start
    source_path: str
    width: int
    height: int
    fps: float


class VideoSource:
    """
    Unified video source for files, RTSP streams, and webcams.

    Supports context manager protocol and iteration.

    Example:
        with VideoSource("data/raw/scene.mp4") as src:
            for frame, meta in src.read_frames():
                ...

        # RTSP camera
        with VideoSource("rtsp://192.168.1.100:554/stream") as src:
            for frame, meta in src.read_frames():
                ...

        # Webcam
        with VideoSource(0) as src:
            for frame, meta in src.read_frames():
                ...
    """

    def __init__(
        self,
        source: Union[str, int],
        resize: Optional[tuple] = None,
        buffer_size: int = 4,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0,
    ):
        self.source            = source
        self.resize            = resize
        self.buffer_size       = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay   = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx   = 0
        self._t_start     = 0.0
        self._source_str  = str(source)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and Path(self.source).exists()

    @property
    def is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.startswith("rtsp://")

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0

    @property
    def total_frames(self) -> int:
        if self._cap and self.is_file:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1   # unknown for streams

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    def open(self):
        """Open the video source with retry logic."""
        for attempt in range(1, self.reconnect_attempts + 1):
            if self.is_rtsp:
                # RTSP: set buffer size to reduce latency
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(self.source)

            if cap.isOpened():
                self._cap     = cap
                self._t_start = time.time()
                self._frame_idx = 0
                logger.info(
                    f"VideoSource opened: {self._source_str} | "
                    f"{self.width}x{self.height} @ {self.fps:.1f}fps"
                )
                return
            logger.warning(
                f"Attempt {attempt}/{self.reconnect_attempts}: "
                f"Cannot open {self._source_str}"
            )
            time.sleep(self.reconnect_delay)
        raise RuntimeError(f"Failed to open video source: {self._source_str}")

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info(f"VideoSource released: {self._source_str}")

    # ── Frame reading ─────────────────────────────────────────────────────

    def read(self) -> tuple[bool, Optional[cv2.Mat]]:
        """Read a single frame. Returns (success, frame)."""
        if not self._cap:
            return False, None
        ret, frame = self._cap.read()
        if ret:
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            self._frame_idx += 1
        return ret, frame

    def read_frames(
        self,
        skip: int = 0,
        max_frames: Optional[int] = None,
    ) -> Iterator[tuple[cv2.Mat, FrameMeta]]:
        """
        Generator: yields (frame, metadata) tuples.

        Args:
            skip:       yield every (skip+1)-th frame
            max_frames: stop after this many frames

        Yields:
            (BGR frame ndarray, FrameMeta)
        """
        if not self._cap:
            self.open()

        count = 0
        internal_idx = 0

        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                # Attempt reconnect for streams
                if (self.is_rtsp or self.is_webcam) and self._try_reconnect():
                    continue
                break

            internal_idx += 1
            if skip > 0 and (internal_idx - 1) % (skip + 1) != 0:
                continue

            meta = FrameMeta(
                frame_idx   = self._frame_idx,
                timestamp   = time.time() - self._t_start,
                source_path = self._source_str,
                width       = frame.shape[1],
                height      = frame.shape[0],
                fps         = self.fps,
            )
            yield frame, meta
            count += 1
            if max_frames and count >= max_frames:
                break

    def _try_reconnect(self) -> bool:
        logger.warning("Stream dropped. Attempting reconnect...")
        self.release()
        time.sleep(self.reconnect_delay)
        try:
            self.open()
            return True
        except RuntimeError:
            logger.error("Reconnect failed. Stopping.")
            return False

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "VideoSource":
        """
        Build a VideoSource from config dict.
        Reads cfg["io"] for source, resize, buffer_size.
        """
        io_cfg = cfg.get("io", {})
        source = io_cfg.get("source", 0)
        resize = tuple(io_cfg["resize"]) if io_cfg.get("resize") else None
        return cls(
            source            = source,
            resize            = resize,
            buffer_size       = io_cfg.get("buffer_size", 4),
            reconnect_attempts = io_cfg.get("reconnect_attempts", 3),
        )


# ── Threaded Frame Buffer ─────────────────────────────────────────────────

class ThreadedVideoSource(VideoSource):
    """
    VideoSource with a background thread that pre-fetches frames
    into a queue, preventing I/O from blocking the main loop.

    Useful for RTSP and webcam sources where reads can stall.

    Example:
        with ThreadedVideoSource("rtsp://...") as src:
            for frame, meta in src.read_frames():
                ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def open(self):
        super().open()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._fill_queue, daemon=True, name="FrameBuffer"
        )
        self._thread.start()

    def release(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        super().release()

    def _fill_queue(self):
        while not self._stop_event.is_set():
            ret, frame = super().read()
            if not ret:
                self._queue.put(None)   # sentinel
                break
            try:
                self._queue.put((frame, self._frame_idx), timeout=1)
            except queue.Full:
                pass   # drop oldest implicitly via maxsize

    def read(self):
        try:
            item = self._queue.get(timeout=2)
            if item is None:
                return False, None
            frame, idx = item
            self._frame_idx = idx
            return True, frame
        except queue.Empty:
            return False, None


# ── Output Video Writer ───────────────────────────────────────────────────

class AnnotatedVideoWriter:
    """
    Wraps cv2.VideoWriter with automatic metadata overlay and
    convenience methods.

    Example:
        with AnnotatedVideoWriter("outputs/result.mp4", fps=25, size=(640,480)) as w:
            w.write(annotated_frame)
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        size: tuple,          # (width, height)
        codec: str = "mp4v",
        add_timestamp: bool = True,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer  = cv2.VideoWriter(output_path, fourcc, fps, size)
        self._size    = size
        self._ts      = add_timestamp
        self._t_start = time.time()
        self._count   = 0
        self.output_path = output_path
        logger.info(f"VideoWriter: {output_path} @ {fps}fps {size[0]}x{size[1]}")

    def write(self, frame: cv2.Mat):
        out = cv2.resize(frame, self._size)
        if self._ts:
            elapsed = time.time() - self._t_start
            ts = time.strftime("%H:%M:%S") + f".{int(elapsed*10%10)}"
            cv2.putText(out, ts, (out.shape[1] - 110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        self._writer.write(out)
        self._count += 1

    def release(self):
        self._writer.release()
        logger.info(f"VideoWriter closed: {self._count} frames → {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
