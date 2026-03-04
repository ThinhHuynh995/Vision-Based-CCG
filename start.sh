#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p data/uploads data/processed data/results data/datasets data/features data/models data/reports
python - <<'PY'
import importlib
mods=["fastapi","uvicorn","cv2","numpy","PIL","aiofiles","jinja2"]
missing=[m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("Thiếu dependencies:", ", ".join(missing))
    print("Hãy chạy: pip install -r requirements.txt")
else:
    print("Dependencies OK")
PY
echo "Khởi động DCAD App tại http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
