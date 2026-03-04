"""
tools/label_tool.py — Interactive CCTV Video Labeling Tool
===========================================================

Lets you scrub through video files, extract frames, and assign
behavior labels by pressing number keys. Saves labels to JSON
and extracts cropped images ready for training.

Usage:
    python tools/label_tool.py --video data/raw/scene01.mp4
    python tools/label_tool.py --video data/raw/scene01.mp4 --step 10
    python tools/label_tool.py --batch data/raw/          # label all videos in folder

Controls (OpenCV window):
    SPACE / D   → next frame (by --step)
    A           → previous frame
    0–4         → assign label  (0=Normal 1=Fighting 2=Falling 3=Loitering 4=Panic)
    S           → save current frame as labeled crop
    F           → toggle fast-forward (5x speed)
    Q / ESC     → quit and save session

Output:
    data/labeled/<video_stem>/
        labels.json          ← {frame_idx: label_id}
        Normal/              ← extracted crops per class
        Fighting/
        Falling/
        Loitering/
        Crowd Panic/
"""
from __future__ import annotations
import cv2
import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import get_logger

logger = get_logger("label_tool")

CLASS_NAMES = {
    0: "Normal",
    1: "Fighting",
    2: "Falling",
    3: "Loitering",
    4: "Crowd Panic",
}
CLASS_COLORS = {
    0: (0, 200, 130),
    1: (0,  50, 220),
    2: (0, 165, 255),
    3: (180, 60, 210),
    4: (0,   0, 220),
}


# ── Core Labeler ──────────────────────────────────────────────────────────

class VideoLabeler:
    """
    Interactive per-frame labeling session for a single video.

    Attributes:
        video_path:   path to input video
        out_dir:      root output directory for this video
        step:         frame skip amount per keypress
        labels:       {frame_idx: label_id}
    """

    def __init__(self, video_path: str, out_dir: str, step: int = 5):
        self.video_path = video_path
        self.out_dir    = Path(out_dir)
        self.step       = step
        self.labels: Dict[int, int] = {}
        self._load_existing_labels()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_idx    = 0
        self.current_label: int = 0
        self._fast_forward = False

        # Create class subdirs
        for cls_name in CLASS_NAMES.values():
            (self.out_dir / cls_name).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VideoLabeler: {video_path} | "
            f"{self.total_frames} frames @ {self.fps:.1f}fps | "
            f"step={step}"
        )

    def _load_existing_labels(self):
        label_file = self.out_dir / "labels.json"
        if label_file.exists():
            with open(label_file) as f:
                raw = json.load(f)
            self.labels = {int(k): v for k, v in raw.items()}
            logger.info(f"Resumed session: {len(self.labels)} labels loaded")

    def _save_labels(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        label_file = self.out_dir / "labels.json"
        with open(label_file, "w") as f:
            json.dump(self.labels, f, indent=2)
        logger.info(f"Labels saved → {label_file}  ({len(self.labels)} entries)")

    def _read_frame(self, idx: int) -> Optional[cv2.Mat]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _draw_ui(self, frame: cv2.Mat) -> cv2.Mat:
        """Overlay HUD on frame."""
        h, w = frame.shape[:2]
        out = frame.copy()

        # Top bar
        cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
        ts = f"{int(self.frame_idx / self.fps // 60):02d}:{int(self.frame_idx / self.fps % 60):02d}"
        cv2.putText(out, f"Frame: {self.frame_idx}/{self.total_frames}  [{ts}]",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(out, f"Labeled: {len(self.labels)}  |  Fast: {'ON' if self._fast_forward else 'off'}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Progress bar
        pct = self.frame_idx / max(self.total_frames - 1, 1)
        bar_w = w - 20
        cv2.rectangle(out, (10, 48), (10 + bar_w, 52), (60, 60, 60), -1)
        cv2.rectangle(out, (10, 48), (10 + int(bar_w * pct), 52), (0, 200, 130), -1)

        # Current label badge
        color = CLASS_COLORS[self.current_label]
        label_text = f"[{self.current_label}] {CLASS_NAMES[self.current_label]}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (w - tw - 20, 8), (w - 4, 8 + th + 8), color, -1)
        cv2.putText(out, label_text, (w - tw - 12, 8 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Already labeled marker
        if self.frame_idx in self.labels:
            lbl = self.labels[self.frame_idx]
            c = CLASS_COLORS[lbl]
            cv2.rectangle(out, (0, 0), (6, h), c, -1)

        # Bottom legend
        legend_y = h - 10
        for idx, (k, v) in enumerate(CLASS_NAMES.items()):
            text = f"{k}:{v[:4]}"
            color_bg = CLASS_COLORS[k]
            x = 10 + idx * 100
            cv2.rectangle(out, (x, legend_y - 14), (x + 90, legend_y + 2), color_bg, -1)
            cv2.putText(out, text, (x + 4, legend_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Controls reminder
        cv2.putText(out, "A/D:prev/next  S:save crop  F:fast  Q:quit",
                    (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        return out

    def _save_crop(self, frame: cv2.Mat):
        cls_name = CLASS_NAMES[self.current_label]
        stem = Path(self.video_path).stem
        fname = f"{stem}_f{self.frame_idx:06d}.jpg"
        out_path = self.out_dir / cls_name / fname
        cv2.imwrite(str(out_path), frame)
        self.labels[self.frame_idx] = self.current_label
        logger.info(f"Saved crop → {out_path}  label={cls_name}")

    def run(self):
        """Main interactive labeling loop."""
        win = f"Label Tool — {Path(self.video_path).name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 560)

        frame = self._read_frame(self.frame_idx)
        if frame is None:
            logger.error("Could not read first frame.")
            return

        while True:
            display = self._draw_ui(frame)
            cv2.imshow(win, display)

            delay = 20 if not self._fast_forward else 1
            key = cv2.waitKey(delay) & 0xFF

            # ── Key handling ──────────────────────────────────────────────
            if key in (ord('q'), 27):           # Q / ESC → quit
                break

            elif key in (ord('d'), ord(' ')):   # next
                jump = self.step * (5 if self._fast_forward else 1)
                self.frame_idx = min(self.frame_idx + jump, self.total_frames - 1)
                frame = self._read_frame(self.frame_idx)

            elif key == ord('a'):               # prev
                self.frame_idx = max(self.frame_idx - self.step, 0)
                frame = self._read_frame(self.frame_idx)

            elif key == ord('s'):               # save crop
                self._save_crop(frame)
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]),
                              CLASS_COLORS[self.current_label], 8)
                cv2.imshow(win, display)
                cv2.waitKey(150)

            elif key == ord('f'):               # toggle fast-forward
                self._fast_forward = not self._fast_forward

            elif ord('0') <= key <= ord('4'):   # set label
                self.current_label = key - ord('0')

            if frame is None:
                logger.warning("End of video or read error.")
                break

        cv2.destroyAllWindows()
        self.cap.release()
        self._save_labels()
        self._print_summary()

    def _print_summary(self):
        from collections import Counter
        counts = Counter(self.labels.values())
        print("\n" + "═" * 45)
        print(f"  Labeling Summary: {Path(self.video_path).name}")
        print("═" * 45)
        for cls_id, cls_name in CLASS_NAMES.items():
            n = counts.get(cls_id, 0)
            bar = "█" * min(n, 30)
            print(f"  {cls_id} {cls_name:<14} {bar:30s} {n}")
        print(f"  Total labeled: {len(self.labels)}")
        print("═" * 45 + "\n")

    def export_dataset_stats(self) -> Dict:
        from collections import Counter
        counts = Counter(self.labels.values())
        return {
            "video": str(self.video_path),
            "total_frames": self.total_frames,
            "labeled_frames": len(self.labels),
            "class_counts": {CLASS_NAMES[k]: v for k, v in counts.items()},
            "label_file": str(self.out_dir / "labels.json"),
        }


# ── Batch Labeler ─────────────────────────────────────────────────────────

def batch_label(folder: str, step: int = 5):
    """Label all MP4/AVI videos in a folder sequentially."""
    folder = Path(folder)
    videos = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.avi"))
    if not videos:
        print(f"No MP4/AVI files found in {folder}")
        return

    print(f"Found {len(videos)} videos to label.\n")
    for i, v in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {v.name}")
        out_dir = folder.parent / "labeled" / v.stem
        labeler = VideoLabeler(str(v), str(out_dir), step=step)
        labeler.run()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive CCTV Video Labeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video",  type=str, help="Single video file to label")
    parser.add_argument("--batch",  type=str, help="Folder containing videos to label")
    parser.add_argument("--step",   type=int, default=5,
                        help="Frame step per keypress (default: 5)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output directory (default: data/labeled/<video_stem>)")
    args = parser.parse_args()

    if args.video:
        out_dir = args.out or f"data/labeled/{Path(args.video).stem}"
        labeler = VideoLabeler(args.video, out_dir, step=args.step)
        labeler.run()
    elif args.batch:
        batch_label(args.batch, step=args.step)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ── Image Folder Labeler ──────────────────────────────────────────────────

class ImageFolderLabeler:
    """
    Label a folder of static images (JPG/PNG) — for cameras that save
    individual frames instead of video files.

    Controls (OpenCV window):
        D / SPACE   → next image
        A           → previous image
        0–4         → assign label
        S           → move image to labeled/<class>/ folder
        Q / ESC     → quit and save

    Output:
        data/labeled/<folder_name>/
            Normal/   Fighting/   Falling/   Loitering/   Crowd Panic/
            labels.json
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, folder: str, out_dir: str):
        self.folder  = Path(folder)
        self.out_dir = Path(out_dir)
        self.images  = sorted([
            p for p in self.folder.iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        ])
        if not self.images:
            raise RuntimeError(f"No images found in {folder}")

        self.labels: Dict[str, int] = {}
        self.current_label = 0
        self.idx = 0

        for cls_name in CLASS_NAMES.values():
            (self.out_dir / cls_name).mkdir(parents=True, exist_ok=True)

        logger.info(f"ImageFolderLabeler: {len(self.images)} images in {folder}")

    def _save_labels(self):
        with open(self.out_dir / "labels.json", "w") as f:
            json.dump(self.labels, f, indent=2)

    def _draw_ui(self, frame: cv2.Mat, img_path: Path) -> cv2.Mat:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 46), (20, 20, 20), -1)
        cv2.putText(out, f"{self.idx+1}/{len(self.images)}  {img_path.name}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1)
        cv2.putText(out, f"Labeled: {len(self.labels)}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1)
        # Label badge
        color = CLASS_COLORS[self.current_label]
        text  = f"[{self.current_label}] {CLASS_NAMES[self.current_label]}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (w-tw-20, 8), (w-4, 8+th+8), color, -1)
        cv2.putText(out, text, (w-tw-12, 8+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Already labeled marker
        fname = img_path.name
        if fname in self.labels:
            lbl_id = self.labels[fname]
            cv2.rectangle(out, (0, 0), (6, h), CLASS_COLORS[lbl_id], -1)
        # Bottom hint
        cv2.putText(out, "A/D:prev/next  S:save  0-4:label  Q:quit",
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
        return out

    def run(self):
        win = f"Image Labeler — {self.folder.name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, 520)

        while True:
            img_path = self.images[self.idx]
            frame    = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Cannot read {img_path}")
                self.idx = min(self.idx + 1, len(self.images) - 1)
                continue

            display = self._draw_ui(frame, img_path)
            cv2.imshow(win, display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('d'), ord(' ')):
                self.idx = min(self.idx + 1, len(self.images) - 1)
            elif key == ord('a'):
                self.idx = max(self.idx - 1, 0)
            elif ord('0') <= key <= ord('4'):
                self.current_label = key - ord('0')
            elif key == ord('s'):
                cls_name = CLASS_NAMES[self.current_label]
                dst      = self.out_dir / cls_name / img_path.name
                import shutil
                shutil.copy2(img_path, dst)
                self.labels[img_path.name] = self.current_label
                logger.info(f"Saved {img_path.name} → {cls_name}")
                # Flash green border
                flash = display.copy()
                cv2.rectangle(flash, (0,0), (flash.shape[1], flash.shape[0]),
                              CLASS_COLORS[self.current_label], 10)
                cv2.imshow(win, flash)
                cv2.waitKey(120)
                # Auto-advance
                self.idx = min(self.idx + 1, len(self.images) - 1)

        cv2.destroyAllWindows()
        self._save_labels()
        from collections import Counter
        counts = Counter(self.labels.values())
        print(f"\nLabeled {len(self.labels)}/{len(self.images)} images")
        for cid, cname in CLASS_NAMES.items():
            print(f"  {cname:<14} {counts.get(cid,0)}")


def smart_label(path: str, step: int = 5, out_dir: str = None):
    """
    Auto-detect input type and launch the right labeler:
      - folder of images  → ImageFolderLabeler
      - video file        → VideoLabeler
      - folder of videos  → batch VideoLabeler
    """
    p = Path(path)
    IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff"}
    VID_EXTS = {".mp4",".avi",".mov",".mkv",".ts"}

    if p.is_dir():
        # Check if it contains images or videos
        imgs   = [f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS]
        videos = [f for f in p.iterdir() if f.suffix.lower() in VID_EXTS]
        if imgs and not videos:
            od = out_dir or f"data/labeled/{p.name}"
            ImageFolderLabeler(str(p), od).run()
        elif videos:
            batch_label(str(p), step=step)
        else:
            print(f"No images or videos found in {p}")
    elif p.is_file():
        if p.suffix.lower() in VID_EXTS:
            od = out_dir or f"data/labeled/{p.stem}"
            VideoLabeler(str(p), od, step=step).run()
        elif p.suffix.lower() in IMG_EXTS:
            od = out_dir or f"data/labeled/{p.parent.name}"
            ImageFolderLabeler(str(p.parent), od).run()
        else:
            print(f"Unsupported file type: {p.suffix}")
    else:
        print(f"Path not found: {path}")
