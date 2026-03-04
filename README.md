# 🎥 Vision-Based Phát Hiện Hành Vi Bất Thường & Phân Tích Lưu Lượng Người

> **BT Nhóm — Môn: Thị Giác Máy Tính | HCMUT**

---

## 📁 Cấu Trúc Dự Án

```
vision_project/
├── configs/
│   └── config.yaml              # Tất cả tham số cấu hình
├── src/
│   ├── preprocessing/
│   │   └── image_processor.py   # Branch 1: Lọc nhiễu, CLAHE, Wiener
│   ├── detection/
│   │   └── detector.py          # Branch 2: YOLOv8, HOG, Density Map
│   ├── classification/
│   │   └── behavior_classifier.py  # Branch 3: MobileNetV3, fine-tune
│   ├── augmentation/
│   │   └── augmentor.py         # Branch 4: Albumentations, MixUp, CutMix
│   ├── tracking/
│   │   └── tracker.py           # Branch 5: IoU Tracker, Optical Flow, Anomaly
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       └── visualizer.py
├── demo/
│   └── gradio_app.py            # Web UI (4 tabs)
├── tests/
│   └── test_pipeline.py         # Unit tests cho tất cả modules
├── data/
│   ├── raw/                     # Video CCTV gốc
│   ├── processed/               # Dataset phân loại (train/val per class)
│   └── samples/
├── models/weights/              # Fine-tuned model weights (.pth)
├── outputs/
│   ├── videos/                  # Video đã annotate
│   ├── images/
│   └── reports/                 # Log files, training plots
├── main.py                      # Entry point chính
└── requirements.txt
```

---

## ⚡ Cài Đặt Nhanh

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Chạy demo ngay (không cần dataset)
python main.py --mode demo
```

---

## 🚀 Các Chế Độ Chạy

### Demo (không cần dataset)
```bash
python main.py --mode demo
# Tạo synthetic video + chạy full pipeline
# Output: outputs/videos/demo_output.mp4
```

### Xử lý video thực
```bash
python main.py --mode video --input data/raw/my_cctv.mp4
# Output: outputs/videos/my_cctv_analyzed.mp4
```

### Augment dataset
```bash
python main.py --mode augment --input data/processed --output data/augmented
```

### Train classifier
```bash
# Chuẩn bị dataset trước:
# data/processed/train/Normal/*.jpg
# data/processed/train/Fighting/*.jpg
# data/processed/train/Falling/*.jpg
# data/processed/train/Loitering/*.jpg
# data/processed/train/Crowd Panic/*.jpg
# (tương tự cho val/)

python main.py --mode train
# Weights lưu tại: models/weights/behavior_clf.pth
```

### Gradio Web UI
```bash
python main.py --mode gradio
# Mở http://localhost:7860
```

---

## 🧪 Chạy Unit Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📋 5 Nhánh Kỹ Thuật

| Nhánh | File | Kỹ thuật |
|-------|------|----------|
| **1. Image Restoration** | `src/preprocessing/image_processor.py` | Gaussian/Median/Bilateral/NLM Denoise, CLAHE, Gamma, Wiener Deblur |
| **2. Detection/Segmentation** | `src/detection/detector.py` | YOLOv8n (fallback: HOG), MOG2 Background Subtraction, Gaussian Density Map |
| **3. Classification** | `src/classification/behavior_classifier.py` | MobileNetV3/ResNet-50/EfficientNet-B0, Transfer Learning, 5 lớp hành vi |
| **4. Data Augmentation** | `src/augmentation/augmentor.py` | Albumentations pipeline, MixUp, CutMix, Batch generator |
| **5. Video Tracking** | `src/tracking/tracker.py` | IoU Tracker, Farneback Optical Flow, Anomaly Detection rule-based |

---

## 🎯 Classes Hành Vi

| ID | Tên | Mô tả |
|----|-----|-------|
| 0 | Normal | Di chuyển bình thường |
| 1 | Fighting | Đánh nhau, xô xát |
| 2 | Falling | Ngã, té ngã |
| 3 | Loitering | Đứng/đi lại vô mục đích lâu |
| 4 | Crowd Panic | Đám đông chạy loạn |

---

## 📊 Metrics Đánh Giá

- **Detection**: mAP@0.5, Precision, Recall
- **Classification**: Accuracy, F1-Score per class, Confusion Matrix
- **Tracking**: MOTA, MOTP (nếu có ground truth)
- **Preprocessing**: PSNR, SSIM (trước/sau denoising)

---

## 🔧 Tuỳ Chỉnh

Chỉnh `configs/config.yaml` để:
- Đổi model YOLO: `detection.model: yolov8s`
- Đổi backbone: `classification.model: efficientnet_b0`
- Bật GPU: `device: cuda`
- Điều chỉnh ngưỡng anomaly: `anomaly.speed_threshold`, `anomaly.loiter_frames`

---

## 📦 Dataset Gợi Ý

| Dataset | Link | Dùng cho |
|---------|------|----------|
| UCF-Crime | [kaggle](https://www.kaggle.com/datasets) | Classification |
| VIRAT | [viratdata.org](http://viratdata.org) | Video tracking |
| MOT Challenge | [motchallenge.net](https://motchallenge.net) | Tracking eval |
| COCO (person) | [cocodataset.org](https://cocodataset.org) | Detection pretraining |
