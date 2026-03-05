#!/usr/bin/env python3
"""
main.py — Vision-Based Phát Hiện Hành Vi Bất Thường
=====================================================
CLI đầy đủ cho toàn bộ workflow từ thu thập đến inference.

WORKFLOW CHUẨN (từ đầu đến cuối):
──────────────────────────────────
  Bước 1 – Gán nhãn video CCTV:
      python main.py label --video data/raw/cctv.mp4
      python main.py label --batch data/raw/           # nhiều video

  Bước 2 – Build dataset từ ảnh gán nhãn:
      python main.py build
      python main.py build --balance 300               # oversample đến 300/class

  Bước 3 – Kiểm tra dataset:
      python main.py validate

  Bước 4 – Augment dữ liệu:
      python main.py augment

  Bước 5 – Train model:
      python main.py train

  Bước 6 – Chạy inference:
      python main.py run --input data/raw/test.mp4
      python main.py run --input 0                     # webcam
      python main.py run --input rtsp://192.168.1.5:554/stream

  Bước 7 – Demo nhanh (không cần dataset):
      python main.py demo

THAM SỐ CHUNG:
    --config PATH    Đường dẫn config (default: configs/config.yaml)

Chạy `python main.py <lệnh> --help` để xem chi tiết từng lệnh.
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.utils.log as _log
from src.utils.config import load

log = _log.get("main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(args):
    return load(args.config)


def _print_run_summary(s: dict):
    print("\n" + "═"*50)
    print("  KẾT QUẢ XỬ LÝ")
    print("═"*50)
    for k, v in s.items():
        if k != "anomaly_log":
            print(f"  {k:<26}: {v}")
    log_entries = s.get("anomaly_log", [])
    if log_entries:
        print(f"\n  Nhật ký cảnh báo ({len(log_entries)} sự kiện):")
        for e in log_entries[:15]:
            print(f"    [{e['time']}] F{e['frame']:05d} | "
                  f"{e['type']:<18} Track#{e['track_id']}")
        if len(log_entries) > 15:
            print(f"    ... và {len(log_entries)-15} sự kiện khác")
    print("═"*50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_demo(args):
    """Tạo video tổng hợp và chạy full pipeline — không cần dataset."""
    import cv2, numpy as np, time
    cfg     = _cfg(args)
    out_dir = Path(cfg.paths.output_videos)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_in  = str(out_dir / "demo_input.mp4")
    demo_out = str(out_dir / "demo_output.mp4")

    log.info("=== DEMO MODE – tạo video tổng hợp ===")
    h, w, fps, n = 480, 640, 15, 200
    wr = cv2.VideoWriter(demo_in, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    persons = [
        {"x": 80,  "y": 230, "vx": 5,   "vy": 0.1,  "col": (80, 200, 100)},
        {"x": 550, "y": 310, "vx": -4,  "vy": -0.2, "col": (80, 140, 220)},
        {"x": 290, "y": 160, "vx": 1.5, "vy": 4,    "col": (210, 100, 80)},
    ]
    bg = np.full((h, w, 3), 42, dtype=np.uint8)
    cv2.rectangle(bg, (0, h-60), (w, h), (62, 62, 62), -1)
    for _ in range(n):
        frame = bg.copy()
        frame = cv2.add(frame, np.random.randint(0, 7, frame.shape, dtype=np.uint8))
        for p in persons:
            p["x"] = (p["x"] + p["vx"]) % w
            p["y"] = float(max(60, min(h-110, p["y"] + p["vy"])))
            cx, cy = int(p["x"]), int(p["y"])
            cv2.rectangle(frame, (cx-14, cy-40), (cx+14, cy+40), p["col"], -1)
            cv2.ellipse(frame, (cx, cy-50), (13,13), 0, 0, 360, p["col"], -1)
        wr.write(frame)
    wr.release()
    log.info(f"Demo input → {demo_in}")

    from src.tracking.tracker import Pipeline
    pipe    = Pipeline(cfg)
    summary = pipe.run(demo_in, demo_out, max_frames=150, show_heatmap=True)
    _print_run_summary(summary)
    log.info(f"Demo output → {demo_out}")


def cmd_label(args):
    """Gán nhãn video CCTV bằng OpenCV labeling tool."""
    use_ai = getattr(args, 'ai', False)
    if use_ai:
        from tools.label_tool import AIAssistedSession
        cfg = _cfg(args)
        if not args.video:
            log.error("--ai chỉ hỗ trợ --video (không hỗ trợ --batch)"); sys.exit(1)
        out = args.out or f"data/labeled/{Path(args.video).stem}"
        AIAssistedSession(args.video, out, step=args.step, cfg=cfg).run()
        return
    from tools.label_tool import LabelSession, batch_label
    if args.batch:
        batch_label(args.batch, step=args.step,
                    out_root=args.out or "data/labeled")
    elif args.video:
        out = args.out or f"data/labeled/{Path(args.video).stem}"
        LabelSession(args.video, out, step=args.step).run()
    else:
        log.error("Cần --video hoặc --batch"); sys.exit(1)


def cmd_build(args):
    """Build dataset từ ảnh gán nhãn → train/val/test split."""
    from src.dataset.builder import DatasetBuilder
    cfg     = _cfg(args)
    ds_cfg  = cfg.dataset
    builder = DatasetBuilder(cfg)
    builder.build(
        labeled_dir       = args.labeled or cfg.paths.data_labeled or "data/labeled",
        output_dir        = args.output  or cfg.paths.data_processed or "data/processed",
        val_ratio         = float(ds_cfg.val_ratio  or 0.15),
        test_ratio        = float(ds_cfg.test_ratio or 0.10),
        target_per_class  = args.balance or (
            int(ds_cfg.target_per_class) if ds_cfg.target_per_class else None
        ),
        remove_duplicates = bool(ds_cfg.remove_duplicates if ds_cfg.remove_duplicates is not None else True),
        min_size          = tuple(ds_cfg.min_image_size) if ds_cfg.min_image_size else (32,32),
    )


def cmd_validate(args):
    """Kiểm tra toàn bộ dataset."""
    from src.dataset.builder import validate
    cfg    = _cfg(args)
    path   = args.input or cfg.paths.data_processed or "data/processed"
    report = validate(path, min_per_class=args.min_samples)
    if not report.passed:
        log.error("Validation FAILED – kiểm tra lỗi ở trên trước khi train")
        sys.exit(1)
    log.info("Validation PASSED ✅")


def cmd_augment(args):
    """Augment dataset – nhân bội ảnh train."""
    from src.augmentation.augmentor import augment_dir
    cfg    = _cfg(args)
    src    = args.input  or str(Path(cfg.paths.data_processed) / "train")
    dst    = args.output or str(Path(cfg.paths.data_processed) / "train_aug")
    n      = args.n or int(cfg.augmentation.n_per_image or 5)
    total  = augment_dir(src, dst, n=n, cfg=cfg.augmentation)
    log.info(f"Augment hoàn tất: {total} ảnh → {dst}")


def cmd_train(args):
    """Fine-tune behavior classifier trên dataset đã build."""
    import torch
    from torch.utils.data import DataLoader
    from src.classification.classifier import (
        BehaviorDataset, build_model, get_transforms, Trainer
    )
    cfg     = _cfg(args)
    clf_cfg = cfg.classification
    device  = args.device or cfg.device or "cpu"
    log.info(f"=== TRAIN | model={clf_cfg.model} device={device} ===")

    data_root = cfg.paths.data_processed
    size      = tuple(clf_cfg.input_size)
    train_ds  = BehaviorDataset(data_root, "train", get_transforms(size, True))
    val_ds    = BehaviorDataset(data_root, "val",   get_transforms(size, False))

    if len(train_ds) == 0:
        log.error(
            "Không tìm thấy ảnh train.\n"
            "  → Chạy: python main.py label → build → augment → train"
        )
        sys.exit(1)

    bs       = int(clf_cfg.batch_size or 16)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    model   = build_model(str(clf_cfg.model), int(clf_cfg.num_classes))
    trainer = Trainer(model, device=device,
                      lr=float(clf_cfg.lr), weight_decay=float(clf_cfg.weight_decay))
    trainer.fit(train_dl, val_dl, epochs=int(clf_cfg.epochs or 25))
    trainer.save("models/weights/behavior_clf.pth")
    trainer.plot_history(cfg.paths.output_reports)


def cmd_run(args):
    """Chạy inference trên video file, webcam, hoặc RTSP stream."""
    from src.tracking.tracker import Pipeline
    cfg    = _cfg(args)

    # Xác định nguồn input
    source = args.input
    if source is None:
        source = cfg.source
    if source is None:
        log.error("Cần --input (video file / 0 cho webcam / rtsp://...)"); sys.exit(1)

    # Thử chuyển sang int cho webcam
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Xác định output path
    if args.output:
        out_path = args.output
    else:
        out_dir = Path(cfg.paths.output_videos)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(source)).stem if isinstance(source, str) else f"webcam_{source}"
        out_path = str(out_dir / f"{stem}_analyzed.mp4")

    log.info(f"=== RUN | source={source} → {out_path} ===")
    pipe    = Pipeline(cfg)
    summary = pipe.run(
        source, out_path,
        max_frames  = args.max_frames,
        show_heatmap = not args.no_heatmap,
        show_flow    = args.flow,
    )
    _print_run_summary(summary)

    # Xuất CSV anomaly log
    if summary["anomaly_log"]:
        import csv
        csv_path = Path(cfg.paths.output_reports) / "anomaly_log.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["time","frame","track_id","type","cx","cy"])
            w.writeheader()
            for e in summary["anomaly_log"]:
                cx, cy = e.get("center", (0,0))
                w.writerow({"time":e["time"],"frame":e["frame"],"track_id":e["track_id"],
                            "type":e["type"],"cx":cx,"cy":cy})
        log.info(f"CSV log → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Parser
# ─────────────────────────────────────────────────────────────────────────────


def cmd_autolabel(args):
    """Tự động gán nhãn video/folder bằng CLIP hoặc model đã train."""
    from src.autolabel.autolabeler import AutoLabeler
    cfg  = _cfg(args)
    mode = args.mode_al or 'auto'
    al   = AutoLabeler.from_config(cfg, mode=mode)

    if not args.input:
        log.error('Cần --input (video file hoặc folder)'); sys.exit(1)

    from pathlib import Path
    p   = Path(args.input)
    out = args.output or ('data/labeled/' + p.stem if p.is_file()
                          else 'data/labeled')
    step = args.step or 10

    if p.is_dir():
        al.label_folder(str(p), out, frame_step=step)
    else:
        al.label_video(str(p), out, frame_step=step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Vision-Based Phát Hiện Hành Vi Bất Thường",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        metavar="PATH", help="Config file (default: configs/config.yaml)")

    sub = parser.add_subparsers(dest="cmd", metavar="<lệnh>")
    sub.required = True

    # ── demo ─────────────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Chạy demo với video tổng hợp (không cần dataset)")

    # ── label ────────────────────────────────────────────────────────────────
    p_label = sub.add_parser("label", help="Gán nhãn hành vi trên video CCTV")
    g = p_label.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", metavar="FILE",   help="Một file video")
    g.add_argument("--batch", metavar="FOLDER", help="Thư mục chứa nhiều video")
    p_label.add_argument("--step", type=int, default=5, help="Bước frame (default: 5)")
    p_label.add_argument("--out",  metavar="DIR",       help="Thư mục output")
    p_label.add_argument("--ai",   action="store_true",
                          help="Bật AI gợi ý nhãn (ENTER để chấp nhận, 0-4 để sửa)")

    # ── build ─────────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Build train/val/test dataset từ ảnh gán nhãn")
    p_build.add_argument("--labeled",  metavar="DIR",  default=None,
                          help="Thư mục ảnh gán nhãn (default: data/labeled)")
    p_build.add_argument("--output",   metavar="DIR",  default=None,
                          help="Thư mục output     (default: data/processed)")
    p_build.add_argument("--balance",  metavar="N",    type=int, default=None,
                          help="Oversample đến N mẫu/class")

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Kiểm tra chất lượng dataset")
    p_val.add_argument("--input",       metavar="DIR", default=None,
                        help="Thư mục dataset (default: data/processed)")
    p_val.add_argument("--min-samples", metavar="N",   type=int, default=20,
                        dest="min_samples",
                        help="Số ảnh tối thiểu/class (default: 20)")

    # ── augment ───────────────────────────────────────────────────────────────
    p_aug = sub.add_parser("augment", help="Augment ảnh train để tăng dataset")
    p_aug.add_argument("--input",  metavar="DIR", default=None,
                        help="Nguồn (default: data/processed/train)")
    p_aug.add_argument("--output", metavar="DIR", default=None,
                        help="Đích  (default: data/processed/train_aug)")
    p_aug.add_argument("--n",      metavar="N",   type=int, default=None,
                        help="Số bản sao/ảnh (default: lấy từ config)")

    # ── autolabel ─────────────────────────────────────────────────────────────
    p_al = sub.add_parser("autolabel",
                           help="Tự động gán nhãn bằng CLIP (zero-shot) hoặc model đã train (semi)")
    p_al.add_argument("--input",   metavar="SRC",  required=True,
                       help="Video file hoặc folder chứa video")
    p_al.add_argument("--output",  metavar="DIR",  default=None,
                       help="Thư mục output (default: data/labeled/<tên_video>)")
    p_al.add_argument("--mode-al", metavar="MODE", default="auto",
                       dest="mode_al",
                       choices=["auto","zero_shot","semi"],
                       help="auto | zero_shot | semi  (default: auto)")
    p_al.add_argument("--step",    metavar="N",    type=int, default=10,
                       help="Lấy mẫu 1 frame mỗi N frames (default: 10)")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune behavior classifier")
    p_train.add_argument("--device", default=None,
                          help="cpu | cuda | mps (default: lấy từ config)")

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Chạy full pipeline trên video/webcam/RTSP")
    p_run.add_argument("--input",      metavar="SRC",  default=None,
                        help="File video / 0 (webcam) / rtsp://...")
    p_run.add_argument("--output",     metavar="FILE", default=None,
                        help="Output video (auto nếu bỏ trống)")
    p_run.add_argument("--max-frames", metavar="N",    type=int, default=None,
                        dest="max_frames", help="Giới hạn số frame xử lý")
    p_run.add_argument("--no-heatmap", action="store_true",
                        help="Tắt heatmap mật độ")
    p_run.add_argument("--flow",       action="store_true",
                        help="Bật optical flow overlay (PiP góc trên trái)")

    return parser


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    handler = {
        "demo":     cmd_demo,
        "label":    cmd_label,
        "build":    cmd_build,
        "validate": cmd_validate,
        "augment":  cmd_augment,
        "train":    cmd_train,
        "run":      cmd_run,
        "autolabel": cmd_autolabel,
    }.get(args.cmd)

    if handler is None:
        parser.print_help(); sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
