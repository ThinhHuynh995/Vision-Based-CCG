# 🎥 Vision-Based Phát Hiện Hành Vi Bất Thường

> **BT Nhóm – Môn: Thị Giác Máy Tính**  
> CPU-only · CCTV thu thập thực tế · Production-ready

---

## ⚡ Cài đặt nhanh

```bash
git clone <repo> && cd vision_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔁 Workflow chuẩn (từ đầu đến cuối)

```bash
# 1️⃣  Gán nhãn video CCTV (OpenCV window)
python main.py label --video data/raw/cctv.mp4
python main.py label --batch data/raw/          # nhiều video cùng lúc

# 2️⃣  Build dataset train/val/test
python main.py build
python main.py build --balance 300              # oversample đến 300 ảnh/class

# 3️⃣  Kiểm tra dataset trước khi train
python main.py validate

# 4️⃣  Augment ảnh train
python main.py augment

# 5️⃣  Train model
python main.py train

# 6️⃣  Chạy inference
python main.py run --input data/raw/test.mp4
python main.py run --input 0                    # webcam
python main.py run --input rtsp://192.168.1.5:554/stream
python main.py run --input video.mp4 --flow     # bật optical flow

# Demo nhanh (không cần dataset)
python main.py demo
```

---

## 📁 Cấu trúc dự án

```
vision_project/
├── main.py                          ← CLI entry point
├── configs/config.yaml              ← Tất cả cấu hình
├── requirements.txt
│
├── src/
│   ├── preprocessing/processor.py  ← Branch 1: Lọc nhiễu, CLAHE, Wiener
│   ├── detection/detector.py       ← Branch 2: YOLOv8, HOG fallback, Density Map
│   ├── classification/classifier.py← Branch 3: MobileNetV3, ResNet, EfficientNet
│   ├── augmentation/augmentor.py   ← Branch 4: Albumentations, MixUp, CutMix
│   ├── tracking/tracker.py         ← Branch 5: IoU Tracker, Optical Flow, Anomaly
│   ├── dataset/builder.py          ← Dataset pipeline + Validator
│   └── utils/
│       ├── config.py               ← YAML loader với dot-access
│       ├── log.py                  ← Logger màu (console + file)
│       └── draw.py                 ← Hàm vẽ bbox, trajectory, heatmap
│
├── tools/
│   └── label_tool.py               ← OpenCV labeling GUI
│
├── data/
│   ├── raw/                        ← Video CCTV gốc (.mp4, .avi)
│   ├── labeled/                    ← Ảnh crop sau khi gán nhãn
│   │   ├── Normal/
│   │   ├── Fighting/
│   │   ├── Falling/
│   │   ├── Loitering/
│   │   └── Crowd Panic/
│   └── processed/                  ← Dataset sau khi build
│       ├── train/{class}/
│       ├── val/{class}/
│       └── test/{class}/
│
├── models/weights/
│   └── behavior_clf.pth            ← Fine-tuned weights (tạo ra sau train)
│
├── outputs/
│   ├── videos/                     ← Video đầu ra đã annotate
│   └── reports/                    ← Log files, training history, anomaly CSV
│
└── tests/
    └── test_all.py                 ← Pytest suite (50+ test cases)
```

---

## 🎮 Phím tắt Labeling Tool

| Phím | Tác dụng |
|------|----------|
| `D` / `SPACE` | Frame tiếp theo |
| `A` | Frame trước |
| `F` | Toggle fast-forward (5×) |
| `0` – `4` | Chọn nhãn hành vi |
| `S` | Lưu crop hiện tại với nhãn đang chọn |
| `Z` | Undo lần lưu gần nhất |
| `I` | In thống kê ra terminal |
| `Q` / `ESC` | Thoát và lưu session |

---

## 📋 5 Nhánh kỹ thuật

| Nhánh | File | Kỹ thuật |
|-------|------|----------|
| **1. Image Restoration** | `src/preprocessing/processor.py` | Gaussian/Median/Bilateral/NLM, CLAHE, Gamma, Wiener deblur |
| **2. Detection** | `src/detection/detector.py` | YOLOv8n (fallback HOG), MOG2 motion gate, Gaussian density map |
| **3. Classification** | `src/classification/classifier.py` | MobileNetV3/EfficientNet-B0/ResNet-50, transfer learning, early stopping |
| **4. Augmentation** | `src/augmentation/augmentor.py` | Albumentations pipeline, MixUp, CutMix, batch generator |
| **5. Tracking + Anomaly** | `src/tracking/tracker.py` | IoU tracker, Farneback optical flow, rule-based anomaly detection |

---

## 🎯 Classes hành vi

| ID | Tên | Phím |
|----|-----|------|
| 0 | Normal | `0` |
| 1 | Fighting | `1` |
| 2 | Falling | `2` |
| 3 | Loitering | `3` |
| 4 | Crowd Panic | `4` |

---

## 🧪 Chạy tests

```bash
pytest tests/ -v
pytest tests/ -v -k "Preprocessing"   # chỉ 1 nhóm
pytest tests/ -v -k "not trainer"     # bỏ qua test nặng
```

---

## ⚙️ Cấu hình

Chỉnh `configs/config.yaml` — không cần sửa code:

```yaml
detection:
  model: "yolov8n"       # đổi sang yolov8s để chính xác hơn
  confidence: 0.45

classification:
  model: "mobilenet_v3_small"   # hoặc efficientnet_b0
  epochs: 25

anomaly:
  speed_threshold: 80    # pixel/frame
  loiter_frames:   60    # frame đứng yên
  crowd_threshold:  6    # số người
```

---

## 📊 Dataset khuyến nghị (tự thu thập)

Mỗi class cần **tối thiểu 100–200 ảnh** để train ổn định.  
Dùng `python main.py label` để gán nhãn từ video CCTV của nhóm.  
Sau đó `build --balance 300` để cân bằng class.
