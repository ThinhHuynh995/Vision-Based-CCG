"""
tools/label_tool.py — Interactive CCTV Video Labeling Tool (OpenCV)
====================================================================

Mở video lên cửa sổ OpenCV, dùng bàn phím để gán nhãn hành vi
từng frame và lưu crop ảnh theo class.

ĐIỀU KHIỂN:
    D / SPACE   → Frame tiếp (theo bước --step)
    A           → Frame trước
    F           → Toggle fast-forward (5× speed)
    0 – 4       → Chọn nhãn hành vi
    S           → Lưu frame/crop hiện tại với nhãn đang chọn
    Z           → Undo lần lưu gần nhất
    I           → Info: in thống kê ra terminal
    Q / ESC     → Thoát và lưu session

OUTPUT:
    data/labeled/<tên_video>/
        labels.json             ← {frame_idx: label_id}
        Normal/                 ← ảnh crop theo class
        Fighting/
        Falling/
        Loitering/
        Crowd Panic/

DÙNG:
    python tools/label_tool.py --video data/raw/cctv.mp4
    python tools/label_tool.py --video data/raw/cctv.mp4 --step 10 --out data/labeled/
    python tools/label_tool.py --batch data/raw/            # gán nhãn hàng loạt
"""
from __future__ import annotations
import cv2
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.utils.log as _log

log = _log.get("label_tool")

CLASS_NAMES = {0:"Normal", 1:"Fighting", 2:"Falling", 3:"Loitering", 4:"Crowd Panic"}
COLORS = {0:(0,200,130), 1:(0,50,220), 2:(0,165,255), 3:(180,60,210), 4:(0,0,220)}


class LabelSession:
    """Quản lý một phiên gán nhãn cho một video."""

    def __init__(self, video_path: str, out_dir: str, step: int = 5):
        self.vpath   = Path(video_path)
        self.out_dir = Path(out_dir)
        self.step    = step
        self.labels: Dict[int, int] = {}
        self._history: List[Tuple[int,int]] = []   # undo stack

        # Tạo thư mục output
        for cls_name in CLASS_NAMES.values():
            (self.out_dir / cls_name).mkdir(parents=True, exist_ok=True)

        # Load session cũ nếu có
        label_file = self.out_dir / "labels.json"
        if label_file.exists():
            with open(label_file, encoding="utf-8") as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
            log.info(f"Tiếp tục session: {len(self.labels)} nhãn đã có")

        # Mở video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")

        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps   = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.idx   = 0
        self.cur_label = 0
        self.fast  = False

        log.info(f"Video: {video_path} | {self.total} frames @ {self.fps:.1f}fps | step={step}")

    # ── IO ────────────────────────────────────────────────────────────────────

    def _read(self, idx: int) -> Optional[cv2.Mat]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        return f if ok else None

    def save_session(self):
        with open(self.out_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)
        log.info(f"Đã lưu {len(self.labels)} nhãn → {self.out_dir/'labels.json'}")

    def save_crop(self, frame: cv2.Mat):
        cls_name = CLASS_NAMES[self.cur_label]
        fname    = f"{self.vpath.stem}_f{self.idx:06d}.jpg"
        path     = self.out_dir / cls_name / fname
        cv2.imwrite(str(path), frame)
        self.labels[self.idx] = self.cur_label
        self._history.append((self.idx, self.cur_label))
        log.info(f"Crop lưu → {path}")

    def undo(self):
        if not self._history:
            return
        ridx, _ = self._history.pop()
        if ridx in self.labels:
            cls_name = CLASS_NAMES[self.labels.pop(ridx)]
            fname    = f"{self.vpath.stem}_f{ridx:06d}.jpg"
            fp       = self.out_dir / cls_name / fname
            if fp.exists(): fp.unlink()
            log.info(f"Undo: xóa frame {ridx} ({cls_name})")

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame: cv2.Mat) -> cv2.Mat:
        out = frame.copy()
        h, w = out.shape[:2]

        # Top bar
        cv2.rectangle(out, (0,0), (w,54), (18,18,18), -1)
        ts = f"{int(self.idx/self.fps//60):02d}:{int(self.idx/self.fps%60):02d}"
        cv2.putText(out, f"Frame {self.idx}/{self.total}  [{ts}]  Labeled:{len(self.labels)}",
                    (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1)
        cv2.putText(out, f"Fast:{'ON' if self.fast else 'off'}  Step:{self.step}",
                    (10,38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150,150,150), 1)

        # Progress bar
        bar_w = w - 20
        pct   = self.idx / max(self.total-1, 1)
        cv2.rectangle(out, (10,50), (10+bar_w,54), (55,55,55), -1)
        cv2.rectangle(out, (10,50), (10+int(bar_w*pct),54), (0,200,130), -1)

        # Current label badge (top-right)
        col  = COLORS[self.cur_label]
        txt  = f"[{self.cur_label}] {CLASS_NAMES[self.cur_label]}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (w-tw-18, 8), (w-4, 8+th+8), col, -1)
        cv2.putText(out, txt, (w-tw-10, 8+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # Left stripe: already labeled in this class?
        if self.idx in self.labels:
            lbl = self.labels[self.idx]
            cv2.rectangle(out, (0,0), (6,h), COLORS[lbl], -1)

        # Bottom legend
        for i, (k,v) in enumerate(CLASS_NAMES.items()):
            x = 10 + i*122
            cv2.rectangle(out, (x, h-28), (x+112, h-6), COLORS[k], -1)
            cv2.putText(out, f"{k}:{v[:7]}", (x+5, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Controls hint
        cv2.putText(out, "A/D:prev/next  S:save  Z:undo  F:fast  Q:quit",
                    (10, h-36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
        return out

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        win = f"Label Tool — {self.vpath.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 580)

        frame = self._read(self.idx)
        if frame is None:
            log.error("Không đọc được frame đầu tiên."); return

        while True:
            cv2.imshow(win, self._draw_hud(frame))
            key = cv2.waitKey(1 if self.fast else 20) & 0xFF

            if key in (ord('q'), 27):                      # Quit
                break
            elif key in (ord('d'), ord(' ')):              # Next
                jump = self.step * (5 if self.fast else 1)
                self.idx = min(self.idx + jump, self.total - 1)
                frame = self._read(self.idx)
            elif key == ord('a'):                          # Prev
                self.idx = max(self.idx - self.step, 0)
                frame = self._read(self.idx)
            elif key == ord('s') and frame is not None:    # Save
                self.save_crop(frame)
                # Flash border
                flash = frame.copy()
                cv2.rectangle(flash, (0,0), (flash.shape[1],flash.shape[0]),
                              COLORS[self.cur_label], 12)
                cv2.imshow(win, self._draw_hud(flash))
                cv2.waitKey(180)
            elif key == ord('z'):                          # Undo
                self.undo()
            elif key == ord('f'):                          # Fast-forward
                self.fast = not self.fast
            elif key == ord('i'):                          # Info
                self._print_stats()
            elif ord('0') <= key <= ord('4'):              # Set label
                self.cur_label = key - ord('0')

            if frame is None:
                log.warning("Hết video hoặc lỗi đọc frame."); break

        cv2.destroyAllWindows()
        self.cap.release()
        self.save_session()
        self._print_stats()

    def _print_stats(self):
        counts = Counter(self.labels.values())
        print("\n" + "─"*48)
        print(f"  {self.vpath.name}  |  {len(self.labels)} frames gán nhãn")
        print("─"*48)
        for cid, cname in CLASS_NAMES.items():
            n   = counts.get(cid, 0)
            bar = "█" * min(n, 32)
            print(f"  {cid} {cname:<14} {bar:<32} {n:4d}")
        print("─"*48 + "\n")

    def summary(self) -> Dict:
        counts = Counter(self.labels.values())
        return {
            "video":         str(self.vpath),
            "total_frames":  self.total,
            "labeled_frames": len(self.labels),
            "class_counts":  {CLASS_NAMES[k]: v for k, v in counts.items()},
        }


def batch_label(folder: str, step: int = 5, out_root: str = "data/labeled"):
    """Gán nhãn lần lượt tất cả video trong folder."""
    folder = Path(folder)
    videos = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.avi")) + \
             sorted(folder.glob("*.MP4")) + sorted(folder.glob("*.AVI"))
    if not videos:
        print(f"Không tìm thấy video MP4/AVI trong {folder}"); return

    print(f"Tìm thấy {len(videos)} video cần gán nhãn.\n")
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {v.name}")
        out = Path(out_root) / v.stem
        sess = LabelSession(str(v), str(out), step=step)
        sess.run()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CCTV Video Labeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", type=str, default=None, help="Một file video")
    parser.add_argument("--batch", type=str, default=None, help="Thư mục chứa nhiều video")
    parser.add_argument("--step",  type=int, default=5,   help="Bước frame (default: 5)")
    parser.add_argument("--out",   type=str, default=None,help="Thư mục output")
    args = parser.parse_args()

    if args.video:
        stem = Path(args.video).stem
        out  = args.out or f"data/labeled/{stem}"
        LabelSession(args.video, out, step=args.step).run()
    elif args.batch:
        batch_label(args.batch, step=args.step,
                    out_root=args.out or "data/labeled")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
