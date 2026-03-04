# DCAD App (Production-ready Academic Demo)

## Tiếng Việt
Ứng dụng web FastAPI mô phỏng đầy đủ pipeline **8 bước** cho phát hiện bất thường và phân tích lưu lượng người, có thêm module training thực tế (Weakly-Supervised + PCA Autoencoder + CCG calibration) chạy thuần NumPy.

### Chạy nhanh
```bash
cd dcad_app
pip install -r requirements.txt
./start.sh
```
Truy cập: `http://localhost:8000`

### Endpoint chính
- `GET /`, `/pipeline`, `/history`, `/about`, `/training`
- `POST /api/process`
- `GET /api/stats`, `GET /api/history`, `DELETE /api/history/{job_id}`
- Training APIs: scan/extract/train/evaluate/sessions/registry/activate/delete

### Ghi chú
- Đây là **simulation production-ready** cho học thuật, chưa dùng deep model GPU thật.
- Khi có GPU, thay `PCAAutoencoder` bằng AE/C3D/Transformer thật và giữ nguyên API.

## English
A production-style FastAPI app for DCAD simulation with full 8-step computer vision pipeline and weakly supervised training module.

- Server-side templates with Vietnamese UX
- SSE streaming for extraction/training progress
- Persistent model registry and active model switching
