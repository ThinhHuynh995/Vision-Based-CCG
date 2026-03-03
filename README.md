# Vision-Based-CCG

Dự án demo Vision-Based CCG cho phát hiện bất thường và phân tích dòng người trong không gian công cộng.

## Tính năng hiện tại
- Demo web FastAPI: upload ảnh, upload video, webcam/camera realtime.
- Hiển thị khung kết quả có tracking ID + nhãn đối tượng.
- Nếu có `data/dcad_behavior_model.pt`, demo sẽ dùng model này để nhận dạng hành vi (`behavior_label`) trong quá trình phân tích.
- Trả về số liệu 6 bước pipeline trong `step_metrics`.
- Train supervised 5 hành vi từ annotation segment:
  - `normal_flow`
  - `abnormal_gathering`
  - `pushing_shoving`
  - `fighting`
  - `vandalism`
- Cache frame đã xử lý theo từng hành vi để tái sử dụng khi train lần sau.

## Cài đặt
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy demo web
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```
Mở `http://127.0.0.1:8000`.

## Train model hành vi
Script: [`scripts/train.py`](data/train.py)

### Dữ liệu video mặc định
- `D:\NWPU-Videos\videos\NWPUCampusDataset\videos\Train`
- `E:\1. ThS\shanghaitech\shanghaitech\training\videos`

### Annotation
- Mặc định: `data/behavior_annotations.json`
- Template: `data/behavior_annotations.example.json`

Nếu file annotation chưa tồn tại, script tự tạo từ template.
Nếu annotation không có segment hợp lệ, script mặc định tự bootstrap một số segment `normal_flow` từ video có sẵn để chạy tiếp.

### Chạy train cơ bản
```powershell
.\.venv\Scripts\python.exe scripts\train.py --annotations data\behavior_annotations.json
```

### Chạy train nhanh hơn
```powershell
.\.venv\Scripts\python.exe scripts\train.py --annotations data\behavior_annotations.json --max-frames 32 --frame-stride 4 --resize-width 416 --epochs 6
```

### Tùy chọn quan trọng
- `--train-dirs ...`: danh sách thư mục video.
- `--annotations ...`: file annotation đầu vào.
- `--annotation-template ...`: template để auto-create annotation.
- `--cache-dir ...`: thư mục cache xử lý.
- `--no-cache`: tắt đọc/ghi cache.
- `--no-bootstrap-if-empty`: tắt auto-bootstrap khi annotation rỗng/không hợp lệ.
- `--checkpoint ...`: đường dẫn checkpoint đầu ra.

## Cache để tái sử dụng khi train lần sau
Mặc định lưu tại `data/processed_behavior_frames` theo cấu trúc:

- `data/processed_behavior_frames/<behavior_label>/<segment_key>/frames/*.jpg`
- `data/processed_behavior_frames/<behavior_label>/<segment_key>/feature.npy`
- `data/processed_behavior_frames/<behavior_label>/<segment_key>/meta.json`

`segment_key` phụ thuộc vào:
- đường dẫn video
- thời điểm cập nhật file video
- label
- `start_frame`, `end_frame`
- tham số xử lý (`max_frames`, `frame_stride`, `resize_width`)

## GPU
Script tự chọn `cuda` nếu khả dụng và dùng mixed precision (`autocast` + `GradScaler`).

## Ghi chú về nhận dạng
- `dcad_behavior_model.pt`: nhận dạng **hành vi** (5 lớp).
- Nhận dạng **đối tượng** trên khung hiện tại dùng OpenCV detector/tracker (`Person`, `MovingObject`, ID tracking).

Kiểm tra nhanh:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
