"""
src/tracking/tracker.py — Branch 5: Video Tracking & Anomaly Detection
=======================================================================
  • Track          – State của một đối tượng đang theo dõi
  • IoUTracker     – Multi-object tracker thuần IoU, không cần GPU
  • FlowAnalyzer   – Farneback optical flow
  • AnomalyChecker – Rule-based anomaly detection từ track state
  • Pipeline       – Kết nối tất cả branches thành pipeline đầu cuối
"""
from __future__ import annotations
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import src.utils.log as _log

log = _log.get(__name__)


# ── Track ─────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    tid: int
    bbox: Tuple[int,int,int,int]
    center: Tuple[int,int]
    label: int = 0
    conf: float = 1.0
    centers: List[Tuple[int,int]] = field(default_factory=list)
    speeds:  List[float]          = field(default_factory=list)
    stationary: int = 0
    age: int = 0
    last_frame: int = 0

    def update(self, bbox, center, label, conf, fidx):
        if self.centers:
            dx, dy = center[0]-self.centers[-1][0], center[1]-self.centers[-1][1]
            spd = (dx**2 + dy**2)**0.5
            self.speeds.append(spd)
            self.stationary = (self.stationary + 1) if spd < 3 else 0
        self.bbox, self.center = bbox, center
        self.label, self.conf  = label, conf
        self.centers.append(center)
        self.age += 1
        self.last_frame = fidx
        if len(self.centers) > 120: self.centers.pop(0)
        if len(self.speeds)  > 120: self.speeds.pop(0)

    @property
    def avg_speed(self) -> float:
        return float(np.mean(self.speeds[-10:])) if self.speeds else 0.0

    def trail(self, n: int = 40) -> List[Tuple[int,int]]:
        return self.centers[-n:]


# ── IoU Tracker ───────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)


class IoUTracker:
    """
    Multi-object tracker thuần IoU – không cần deep features, chạy tốt trên CPU.

    Sử dụng:
        tracker = IoUTracker(cfg.tracking)
        tracks = tracker.update(dets, labels, confs, frame_idx)
        traj   = tracker.trajectories()
    """

    def __init__(self, cfg):
        self._iou_thr  = float(cfg.iou_threshold or 0.30)
        self._max_age  = int(cfg.max_age         or 30)
        self._min_hits = int(cfg.min_hits        or 3)
        self._tracks: Dict[int, Track] = {}
        self._nid = 1

    def update(self, dets, labels=None, confs=None, fidx: int = 0) -> List[Track]:
        labels = labels or [0]*len(dets)
        confs  = confs  or [1.0]*len(dets)
        matched, used = set(), set()

        # Match detections → existing tracks
        for i, det in enumerate(dets):
            best_iou, best_tid = 0.0, None
            for tid, trk in self._tracks.items():
                if tid in used: continue
                v = _iou(det.bbox, trk.bbox)
                if v > best_iou:
                    best_iou, best_tid = v, tid
            if best_iou >= self._iou_thr and best_tid is not None:
                self._tracks[best_tid].update(det.bbox, det.center, labels[i], confs[i], fidx)
                matched.add(i); used.add(best_tid)
            else:
                t = Track(tid=self._nid, bbox=det.bbox, center=det.center,
                          label=labels[i], conf=confs[i])
                t.centers.append(det.center); t.last_frame = fidx
                self._tracks[self._nid] = t
                self._nid += 1

        # Remove stale
        stale = [tid for tid, t in self._tracks.items()
                 if fidx - t.last_frame > self._max_age]
        for tid in stale:
            del self._tracks[tid]

        return [t for t in self._tracks.values() if t.age >= self._min_hits]

    def trajectories(self) -> Dict[int, List[Tuple[int,int]]]:
        return {tid: t.trail() for tid, t in self._tracks.items()}

    def reset(self):
        self._tracks.clear(); self._nid = 1


# ── Optical Flow ──────────────────────────────────────────────────────────────

class FlowAnalyzer:
    """Farneback dense optical flow cho phát hiện hướng & tốc độ chuyển động."""

    def analyze(self, prev: np.ndarray, curr: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((*g1.shape, 3), dtype=np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── Anomaly Checker ───────────────────────────────────────────────────────────

class AnomalyChecker:
    """
    Rule-based anomaly detection kết hợp classifier label + vật lý track.

    Sử dụng:
        ac = AnomalyChecker(cfg.anomaly)
        anom = ac.check(track)      # str hoặc None
        crowd = ac.check_crowd(n)   # str hoặc None
    """

    def __init__(self, cfg):
        self._spd    = float(cfg.speed_threshold  or 80)
        self._loiter = int(cfg.loiter_frames      or 60)
        self._crowd  = int(cfg.crowd_threshold    or 6)
        self.log: List[Dict] = []

    def check(self, t: Track, fidx: int) -> Optional[str]:
        label_anom = {1: "FIGHTING", 2: "FALLING", 4: "CROWD PANIC"}
        if t.label in label_anom:
            return label_anom[t.label]
        if t.avg_speed > self._spd:
            return "RUNNING FAST"
        if t.label == 3 or t.stationary >= self._loiter:
            return "LOITERING"
        return None

    def check_crowd(self, n: int) -> Optional[str]:
        return f"CROWD DENSITY ({n})" if n >= self._crowd else None

    def record(self, atype: str, tid: int, fidx: int, center: Tuple):
        e = {"time": time.strftime("%H:%M:%S"), "frame": fidx,
             "track_id": tid, "type": atype, "center": center}
        self.log.append(e)
        log.warning(f"⚠  [{atype}] track={tid} frame={fidx}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class Pipeline:
    """
    Kết nối tất cả 5 branches thành pipeline đầu-cuối.

    Sử dụng:
        pipe = Pipeline(cfg)
        summary = pipe.run("data/raw/cctv.mp4", "outputs/videos/out.mp4")
    """

    def __init__(self, cfg):
        from src.preprocessing.processor import Preprocessor
        from src.detection.detector import PersonDetector, DensityMap
        from src.classification.classifier import BehaviorClassifier

        self._pp     = Preprocessor(cfg.preprocessing)
        self._det    = PersonDetector(cfg.detection)
        self._clf    = BehaviorClassifier(cfg.classification)
        self._track  = IoUTracker(cfg.tracking)
        self._anomaly = AnomalyChecker(cfg.anomaly)
        self._flow   = FlowAnalyzer() if cfg.tracking.optical_flow else None
        self._dm: Optional[DensityMap] = None
        self._DensityMap = DensityMap
        log.info("Pipeline sẵn sàng")

    def run(
        self,
        source,
        output_path: str,
        max_frames: Optional[int] = None,
        show_heatmap: bool = True,
        show_flow: bool = False,
    ) -> Dict:
        import src.utils.draw as draw

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        self._dm = self._DensityMap((h, w))
        prev, fidx, n_anom, t0 = None, 0, 0, time.time()

        log.info(f"Bắt đầu xử lý: {source}")

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and fidx >= max_frames):
                break

            # Branch 1: Preprocess
            clean = self._pp(frame)

            # Branch 2: Detect
            dets = self._det(clean)
            self._dm.update(dets)

            # Branch 3: Classify crops
            labels, confs = [], []
            for d in dets:
                crop = d.crop(clean)
                lbl, conf = self._clf(crop) if crop.size > 0 else (0, 1.0)
                labels.append(lbl); confs.append(conf)

            # Branch 5: Track
            tracks = self._track.update(dets, labels, confs, fidx)

            # Anomaly check
            alarms = []
            for t in tracks:
                a = self._anomaly.check(t, fidx)
                if a:
                    self._anomaly.record(a, t.tid, fidx, t.center)
                    alarms.append(a); n_anom += 1
            ca = self._anomaly.check_crowd(len(dets))
            if ca:
                alarms.append(ca)

            # Draw
            out = clean.copy()
            if show_heatmap:
                out = draw.heatmap(out, self._dm.get())
            if tracks:
                out = draw.boxes(out, [t.bbox for t in tracks],
                                 [t.tid for t in tracks],
                                 [t.label for t in tracks],
                                 [t.conf for t in tracks])
            out = draw.trajectories(out, self._track.trajectories())

            # Branch 4 optical flow overlay (small PiP)
            if show_flow and prev is not None and self._flow:
                _, fvis = self._flow.analyze(prev, clean)
                pip = cv2.resize(fvis, (w//4, h//4))
                out[0:h//4, 0:w//4] = pip

            if alarms:
                out = draw.alert_banner(out, " | ".join(set(alarms)))

            out = draw.stats_panel(out, {
                "Frame":   str(fidx),
                "Persons": str(len(dets)),
                "Tracks":  str(len(tracks)),
                "Alerts":  str(n_anom),
            })

            writer.write(out)
            prev = clean.copy()
            fidx += 1

        cap.release(); writer.release()
        elapsed = time.time() - t0

        return {
            "frames_processed": fidx,
            "total_anomalies":  n_anom,
            "fps_achieved":     round(fidx / max(elapsed, 0.001), 1),
            "elapsed_sec":      round(elapsed, 2),
            "anomaly_log":      self._anomaly.log,
        }
