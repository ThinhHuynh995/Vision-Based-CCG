# Vision-Based-CCG

Dự án khởi tạo cho đề tài **Vision-Based Abnormal Behavior Detection and People Flow Analysis in Public Spaces**.

## Kiến trúc
- **FastAPI** phục vụ API và giao diện trực quan tiếng Việt qua `templates`.
- **OpenCV** cho tiền xử lý video và trích xuất đặc trưng chuyển động.
- **PyTorch** cho khung DCAD + Crowd Context Gate.
- Dữ liệu mô tả đề tài và workflow được lưu tại thư mục `/data` dưới dạng JSON.

## Chạy nhanh
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Mở: `http://127.0.0.1:8000`

## Huấn luyện demo
```bash
python scripts/train.py
```

## Điểm mới đề tài (theo triển khai)
1. Gating bất thường theo mật độ đám đông thời gian thực bằng CCG.
2. Ghép density estimation + anomaly detection vào cùng pipeline DCAD.
3. Dashboard hóa quy trình học phần: Input → Augment → Tiền xử lý → Training/Inference → Output.
