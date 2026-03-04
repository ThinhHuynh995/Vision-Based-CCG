"""
main.py — Vision-Based Abnormal Behavior Detection System
==========================================================
Production-grade entry point with full CLI.

Modes:
    demo        Run on synthetic video (no dataset needed)
    video       Process a single video file
    stream      Process RTSP / webcam stream
    train       Fine-tune behavior classifier
    augment     Augment dataset
    build       Build dataset from labeled CCTV footage
    validate    Validate dataset integrity
    label       Launch interactive labeling tool
    report      Generate PDF/CSV report from JSON results
    gradio      Launch Gradio web UI

Examples:
    python main.py --mode demo
    python main.py --mode video --input data/raw/cctv.mp4 --report
    python main.py --mode stream --input rtsp://192.168.1.100:554/stream
    python main.py --mode stream --input 0
    python main.py --mode label  --input data/raw/
    python main.py --mode build  --labeled data/labeled --output data/processed
    python main.py --mode train
    python main.py --mode validate --input data/processed
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("main")


def _make_output_path(input_path, cfg, suffix="_analyzed"):
    p = Path(input_path)
    out_dir = Path(cfg["paths"]["output_videos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{p.stem}{suffix}{p.suffix}")


def _process_and_report(input_path, output_path, cfg,
                         do_report=False, max_frames=None,
                         show_heatmap=True, show_flow=False):
    from src.tracking.tracker import VideoProcessor
    from src.reporting.reporter import Reporter, RunSummary

    vp = VideoProcessor(cfg)
    summary_dict = vp.process_video(
        input_path, output_path,
        show_heatmap=show_heatmap, show_flow=show_flow,
        max_frames=max_frames,
    )
    _print_summary(summary_dict)

    if do_report:
        run = RunSummary(
            video_path=input_path,
            frames_processed=summary_dict["frames_processed"],
            fps_achieved=summary_dict["fps_achieved"],
            elapsed_sec=summary_dict["elapsed_sec"],
            anomaly_log=summary_dict["anomaly_log"],
            config_snapshot=cfg,
        )
        reporter = Reporter(run, cfg["paths"]["output_reports"])
        csv_p  = reporter.export_csv()
        json_p = reporter.export_json()
        pdf_p  = reporter.export_pdf()
        print(f"\n  Reports:\n    CSV  → {csv_p}\n    JSON → {json_p}\n    PDF  → {pdf_p}")
    return summary_dict


def _print_summary(summary):
    print("\n" + "="*52)
    print("  PROCESSING SUMMARY")
    print("="*52)
    for k, v in summary.items():
        if k != "anomaly_log":
            print(f"  {k:<24}: {v}")
    log = summary.get("anomaly_log", [])
    if log:
        print(f"\n  Anomaly Log ({len(log)} events):")
        for ev in log[:12]:
            print(f"    [{ev['time']}] F{ev['frame']:05d} | {ev['type']:<16} Track#{ev['track_id']}")
        if len(log) > 12:
            print(f"    ... and {len(log)-12} more")
    print("="*52 + "\n")


def run_demo(cfg):
    import cv2, numpy as np
    out_dir = Path(cfg["paths"]["output_videos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_in  = str(out_dir / "demo_input.mp4")
    demo_out = str(out_dir / "demo_output.mp4")
    logger.info("=== DEMO MODE ===")
    h, w, fps, n = 480, 640, 15, 180
    writer = cv2.VideoWriter(demo_in, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    persons = [
        {"x": 80,  "y": 220, "vx": 5,  "vy": 0.2,  "col": (90, 200, 120)},
        {"x": 550, "y": 300, "vx": -4, "vy": -0.3,  "col": (80, 140, 210)},
        {"x": 280, "y": 150, "vx": 1,  "vy": 4,     "col": (200, 100, 80)},
    ]
    bg = np.full((h, w, 3), 45, dtype=np.uint8)
    cv2.rectangle(bg, (0, h-55), (w, h), (65, 65, 65), -1)
    for f in range(n):
        frame = bg.copy()
        frame += np.random.randint(0, 6, frame.shape, dtype=np.uint8)
        for p in persons:
            p["x"] = (p["x"] + p["vx"]) % w
            p["y"] = float(max(60, min(h-100, p["y"] + p["vy"])))
            cx, cy = int(p["x"]), int(p["y"])
            cv2.rectangle(frame, (cx-14, cy-38), (cx+14, cy+38), p["col"], -1)
            cv2.ellipse(frame, (cx, cy-48), (13, 13), 0, 0, 360, p["col"], -1)
        writer.write(frame)
    writer.release()
    _process_and_report(demo_in, demo_out, cfg, do_report=True, max_frames=150)
    logger.info(f"Demo output → {demo_out}")


def run_video(cfg, input_path, output_path, do_report, max_frames, heatmap, flow):
    if not Path(input_path).exists():
        logger.error(f"File not found: {input_path}"); sys.exit(1)
    if not output_path:
        output_path = _make_output_path(input_path, cfg)
    logger.info(f"=== VIDEO MODE: {input_path} ===")
    _process_and_report(input_path, output_path, cfg,
                        do_report=do_report, max_frames=max_frames,
                        show_heatmap=heatmap, show_flow=flow)


def run_stream(cfg, source, output_path, do_report, max_frames):
    src = int(source) if source.isdigit() else source
    if not output_path:
        stem = "stream_" + str(source).replace("/","_").replace(":","_")
        output_path = str(Path(cfg["paths"]["output_videos"]) / f"{stem}.mp4")
    logger.info(f"=== STREAM MODE: {src} ===")
    _process_and_report(src, output_path, cfg,
                        do_report=do_report, max_frames=max_frames)


def run_train(cfg):
    import torch
    from torch.utils.data import DataLoader
    from src.classification.behavior_classifier import (
        BehaviorDataset, build_model, get_transforms, Trainer
    )
    device  = cfg.get("device", "cpu")
    clf_cfg = cfg["classification"]
    logger.info(f"=== TRAIN MODE | device={device} ===")
    train_ds = BehaviorDataset(cfg["paths"]["data_processed"], "train",
                                get_transforms(tuple(clf_cfg["input_size"]), True))
    val_ds   = BehaviorDataset(cfg["paths"]["data_processed"], "val",
                                get_transforms(tuple(clf_cfg["input_size"]), False))
    if len(train_ds) == 0:
        logger.error("No training data. Run: label → build → augment → train")
        return
    train_dl = DataLoader(train_ds, batch_size=clf_cfg["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=clf_cfg["batch_size"], shuffle=False, num_workers=0)
    model    = build_model(clf_cfg["model"], clf_cfg["num_classes"])
    trainer  = Trainer(model, device=device, lr=clf_cfg["lr"], weight_decay=clf_cfg["weight_decay"])
    history  = trainer.fit(train_dl, val_dl, epochs=clf_cfg["epochs"])
    trainer.save("models/weights/behavior_clf.pth")


def run_augment(cfg, src_dir, dst_dir):
    from src.augmentation.augmentor import generate_augmented_dataset
    logger.info(f"=== AUGMENT: {src_dir} → {dst_dir} ===")
    generate_augmented_dataset(src_dir, dst_dir, n_per_image=5, cfg=cfg.get("augmentation", {}))


def run_build(cfg, labeled_dir, output_dir, balance, val_ratio, test_ratio):
    from src.dataset.builder import DatasetBuilder
    logger.info(f"=== BUILD: {labeled_dir} → {output_dir} ===")
    DatasetBuilder(cfg).build(labeled_dir=labeled_dir, output_dir=output_dir,
                               val_ratio=val_ratio, test_ratio=test_ratio,
                               target_per_class=balance)


def run_validate(cfg, dataset_path):
    from src.dataset.validator import DatasetValidator
    logger.info(f"=== VALIDATE: {dataset_path} ===")
    report = DatasetValidator.run(dataset_path)
    report.print()
    DatasetValidator.class_distribution_chart(dataset_path)
    if not report.passed:
        sys.exit(1)


def run_label(input_path, step, out_dir=None):
    from tools.label_tool import smart_label
    logger.info(f"=== LABEL MODE: {input_path} ===")
    smart_label(input_path, step=step, out_dir=out_dir)



def run_ingest(cfg, src_root, out_root):
    from src.dataset.builder import ImageFolderIngestor
    src = src_root or cfg.get('ingest',{}).get('src_root','data/raw_images')
    dst = out_root or cfg.get('ingest',{}).get('out_root','data/labeled/static')
    resize = cfg.get('ingest',{}).get('resize', None)
    resize = tuple(resize) if resize else None
    logger.info(f'=== INGEST MODE: {src} → {dst} ===')
    counts = ImageFolderIngestor().ingest(src, dst, resize=resize)
    print('Ingested:', counts)

def run_report(json_path, output_dir):
    import json as _json
    from src.reporting.reporter import Reporter, RunSummary
    with open(json_path) as f:
        data = _json.load(f)
    run = RunSummary(
        video_path=data.get("video_path", ""),
        frames_processed=data.get("frames_processed", 0),
        fps_achieved=data.get("fps_achieved", 0),
        elapsed_sec=data.get("elapsed_sec", 0),
        anomaly_log=data.get("anomaly_log", []),
    )
    reporter = Reporter(run, output_dir)
    reporter.export_csv(); reporter.export_json(); reporter.export_pdf()
    print(f"Reports saved to: {output_dir}")


def run_gradio(cfg):
    try:
        import gradio
    except ImportError:
        logger.error("pip install gradio"); return
    from demo.gradio_app import create_interface
    logger.info("=== GRADIO — http://localhost:7860 ===")
    create_interface(cfg).launch(share=False)


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Based Abnormal Behavior Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True,
        choices=["demo","video","stream","train","augment","build","ingest",
                 "validate","label","report","gradio"])
    parser.add_argument("--input",      type=str,   default=None)
    parser.add_argument("--output",     type=str,   default=None)
    parser.add_argument("--config",     type=str,   default="configs/config.yaml")
    parser.add_argument("--max-frames", type=int,   default=None)
    parser.add_argument("--heatmap",    action="store_true", default=True)
    parser.add_argument("--flow",       action="store_true", default=False)
    parser.add_argument("--report",     action="store_true", default=False)
    parser.add_argument("--labeled",    type=str,   default="data/labeled")
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--balance",    type=int,   default=None)
    parser.add_argument("--step",       type=int,   default=5)
    args = parser.parse_args()
    cfg  = load_config(args.config)

    dispatch = {
        "demo":     lambda: run_demo(cfg),
        "video":    lambda: run_video(cfg, args.input or "", args.output or "",
                                      args.report, args.max_frames, args.heatmap, args.flow),
        "stream":   lambda: run_stream(cfg, args.input or "0", args.output,
                                        args.report, args.max_frames),
        "train":    lambda: run_train(cfg),
        "augment":  lambda: run_augment(cfg, args.input or "data/processed/train",
                                         args.output or "data/augmented"),
        "build":    lambda: run_build(cfg, args.labeled,
                                       args.output or "data/processed",
                                       args.balance, args.val_ratio, args.test_ratio),
        "validate": lambda: run_validate(cfg, args.input or "data/processed"),
        "label":    lambda: run_label(args.input or "data/raw", args.step),
        "ingest":   lambda: run_ingest(cfg, args.input, args.output),
        "report":   lambda: run_report(args.input or "",
                                        args.output or cfg["paths"]["output_reports"]),
        "gradio":   lambda: run_gradio(cfg),
    }
    dispatch[args.mode]()

if __name__ == "__main__":
    main()
