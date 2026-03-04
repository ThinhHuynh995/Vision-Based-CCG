"""
demo/gradio_app.py — Gradio web interface for the detection system.

Provides:
  • Tab 1: Upload image → preprocessing comparison
  • Tab 2: Upload image → person detection + behavior classification
  • Tab 3: Upload short video → full pipeline analysis
  • Tab 4: Augmentation preview
"""
from __future__ import annotations
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path


def create_interface(cfg: dict):
    """Build and return Gradio Blocks interface."""
    import gradio as gr
    from src.preprocessing.image_processor import ImagePreprocessor
    from src.detection.detector import PersonDetector, DensityEstimator
    from src.classification.behavior_classifier import BehaviorClassifier, CLASS_NAMES
    from src.augmentation.augmentor import ImageAugmentor
    from src.utils.visualizer import draw_detections, build_heatmap

    preprocessor = ImagePreprocessor(cfg.get("preprocessing", {}))
    detector     = PersonDetector(cfg.get("detection", {}))
    classifier   = BehaviorClassifier(cfg.get("classification", {}))
    augmentor    = ImageAugmentor(cfg.get("augmentation", {}))

    # ── Tab functions ─────────────────────────────────────────────────────

    def preprocess_image(image_np: np.ndarray, method: str, enhance: str):
        if image_np is None:
            return None
        cfg_override = dict(cfg.get("preprocessing", {}))
        cfg_override["denoise"] = {"method": method, "kernel_size": 5}
        cfg_override["enhancement"] = {"method": enhance, "clip_limit": 2.0, "tile_grid": [8, 8]}
        pp = ImagePreprocessor(cfg_override)
        comparison = pp.compare(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

    def detect_and_classify(image_np: np.ndarray):
        if image_np is None:
            return None, "No image provided."
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        clean = preprocessor.process(frame)
        detections = detector.detect(clean)

        boxes, labels, confs = [], [], []
        results_text = f"Found {len(detections)} person(s):\n\n"

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            crop = clean[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                lbl, conf = classifier.predict(crop)
            else:
                lbl, conf = 0, 1.0
            boxes.append(det.bbox)
            labels.append(lbl)
            confs.append(conf)
            cls_name = CLASS_NAMES.get(lbl, "Unknown")
            results_text += f"Person {i+1}: {cls_name} ({conf*100:.1f}%)\n"

        density = DensityEstimator(frame.shape[:2])
        density.update(detections)
        vis = build_heatmap(clean.copy(), density.get_heatmap())
        vis = draw_detections(vis, boxes, labels=labels, confidences=confs)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), results_text

    def augment_preview(image_np: np.ndarray, n_samples: int):
        if image_np is None:
            return None
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        grid = augmentor.show_grid(frame, n=int(n_samples))
        return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

    def process_video_gradio(video_path: str):
        if not video_path:
            return None, "No video uploaded."
        from src.tracking.tracker import VideoProcessor
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        try:
            vp = VideoProcessor(cfg)
            summary = vp.process_video(video_path, out_path, show_heatmap=True, max_frames=150)
            report = (
                f"Frames processed : {summary['frames_processed']}\n"
                f"Total anomalies  : {summary['total_anomalies']}\n"
                f"Elapsed (s)      : {summary['elapsed_sec']}\n"
                f"FPS achieved     : {summary['fps_achieved']}\n\n"
                "Anomaly Log:\n"
            )
            for ev in summary["anomaly_log"][:20]:
                report += f"  [{ev['time']}] Frame {ev['frame']:04d} | {ev['type']} | Track {ev['track_id']}\n"
            return out_path, report
        except Exception as e:
            return None, f"Error: {e}"

    # ── Build UI ──────────────────────────────────────────────────────────

    with gr.Blocks(title="Vision-Based Behavior Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎥 Vision-Based Phát Hiện Hành Vi Bất Thường\n"
            "> Đề tài BT Nhóm – Thị Giác Máy Tính"
        )

        with gr.Tab("🖼️ Tiền Xử Lý Ảnh"):
            gr.Markdown("**Branch 1** – So sánh ảnh gốc vs sau khi lọc nhiễu & tăng cường")
            with gr.Row():
                img_in = gr.Image(label="Ảnh đầu vào", type="numpy")
                img_out = gr.Image(label="Before | After")
            with gr.Row():
                denoise_dd = gr.Dropdown(
                    ["gaussian", "median", "bilateral", "nlm"],
                    value="gaussian", label="Phương pháp khử nhiễu"
                )
                enhance_dd = gr.Dropdown(
                    ["clahe", "histogram_eq", "gamma"],
                    value="clahe", label="Phương pháp tăng cường"
                )
            btn1 = gr.Button("Xử lý", variant="primary")
            btn1.click(preprocess_image, [img_in, denoise_dd, enhance_dd], img_out)

        with gr.Tab("🚶 Phát Hiện & Phân Loại"):
            gr.Markdown("**Branch 2+3** – Phát hiện người và phân loại hành vi")
            with gr.Row():
                det_in  = gr.Image(label="Ảnh / Frame", type="numpy")
                det_out = gr.Image(label="Kết quả Detection")
            det_txt = gr.Textbox(label="Chi tiết phân loại", lines=6)
            btn2 = gr.Button("Phân tích", variant="primary")
            btn2.click(detect_and_classify, det_in, [det_out, det_txt])

        with gr.Tab("🔁 Data Augmentation"):
            gr.Markdown("**Branch 4** – Xem trước augmentation trên ảnh")
            with gr.Row():
                aug_in  = gr.Image(label="Ảnh gốc", type="numpy")
                aug_out = gr.Image(label="Grid augmented")
            n_slider = gr.Slider(4, 12, value=8, step=4, label="Số mẫu")
            btn3 = gr.Button("Tạo augmented", variant="primary")
            btn3.click(augment_preview, [aug_in, n_slider], aug_out)

        with gr.Tab("📹 Video Pipeline"):
            gr.Markdown("**Branch 5 + Full pipeline** – Xử lý video, tracking, anomaly detection")
            vid_in  = gr.Video(label="Video đầu vào (MP4, tối đa ~30s)")
            btn4    = gr.Button("Phân tích Video", variant="primary")
            vid_out = gr.Video(label="Video kết quả")
            vid_txt = gr.Textbox(label="Báo cáo", lines=12)
            btn4.click(process_video_gradio, vid_in, [vid_out, vid_txt])

    return demo


if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config("configs/config.yaml")
    app = create_interface(cfg)
    app.launch()
