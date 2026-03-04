from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
import threading
import uuid

import cv2
import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline import MediaProcessor
from src.training import Trainer
from src.training.dataset_loader import DatasetLoader
from src.utils import get_stats, list_history

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}

for p in [DATA_DIR, UPLOADS_DIR, DATA_DIR / "processed", RESULTS_DIR, DATA_DIR / "datasets", DATA_DIR / "features", DATA_DIR / "models", DATA_DIR / "reports"]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DCAD System", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
processor = MediaProcessor(DATA_DIR)
trainer = Trainer(DATA_DIR)
loader = DatasetLoader()


def model_status():
    reg = DATA_DIR / "models" / "registry.json"
    if not reg.exists():
        return {"mode": "simulation", "label": "🟡 Simulation Mode"}
    data = json.loads(reg.read_text(encoding="utf-8"))
    active = data.get("active_session")
    if not active:
        return {"mode": "simulation", "label": "🟡 Simulation Mode"}
    s = next((x for x in data.get("sessions", []) if x["session_id"] == active), None)
    if not s:
        return {"mode": "simulation", "label": "🟡 Simulation Mode"}
    auc = s.get("results", {}).get("val_auc", 0)
    return {"mode": "trained", "label": f"🟢 pca_{active[:6]} · AUC {auc*100:.1f}%"}


def ctx(request: Request):
    return {"request": request, "stats": get_stats(DATA_DIR), "history": list_history(RESULTS_DIR, 6), "model_status": model_status()}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", ctx(request))


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return templates.TemplateResponse("pipeline.html", ctx(request))


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    data = ctx(request); data["all_history"] = list_history(RESULTS_DIR, 200)
    return templates.TemplateResponse("history.html", data)


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", ctx(request))


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    d = ctx(request)
    reg = DATA_DIR / "models" / "registry.json"
    d["sessions"] = json.loads(reg.read_text(encoding="utf-8")).get("sessions", []) if reg.exists() else []
    return templates.TemplateResponse("training.html", d)


@app.post("/api/process")
async def api_process(file: UploadFile = File(...), pipeline: str = Form("full"), augment: str = Form("none"), task: str = Form("anomaly")):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Định dạng không hỗ trợ")
    job_id = str(uuid.uuid4())[:8]
    fname = f"{job_id}{ext}"
    out = UPLOADS_DIR / fname
    async with aiofiles.open(out, "wb") as f:
        await f.write(await file.read())
    try:
        return JSONResponse(processor.process(out, fname, job_id, augment=augment, pipeline=pipeline, task=task))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def api_stats():
    return JSONResponse(get_stats(DATA_DIR))


@app.get("/api/history")
async def api_history(limit: int = 20):
    return JSONResponse(list_history(RESULTS_DIR, limit))


@app.delete("/api/history/{job_id}")
async def api_delete_history(job_id: str):
    f = RESULTS_DIR / f"{job_id}.json"
    if not f.exists():
        raise HTTPException(404, "Không tìm thấy job")
    payload = json.loads(f.read_text(encoding="utf-8"))
    for p in payload.get("output_images", {}).values():
        if isinstance(p, str) and p.startswith("/data/"):
            (DATA_DIR / p.replace("/data/", "")).unlink(missing_ok=True)
    f.unlink(missing_ok=True)
    return {"ok": True}


def make_sse_endpoint(blocking_fn):
    async def endpoint(payload: dict):
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run():
            try:
                blocking_fn(payload, progress_cb=lambda d: loop.call_soon_threadsafe(queue.put_nowait, d))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"phase": "error", "msg": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()

        async def gen():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return endpoint


@app.post("/api/training/scan")
async def training_scan(payload: dict):
    root = Path(payload["dataset_path"])
    scan = loader.scan_dataset(root)
    folders = {}
    for v in scan.videos:
        k = v.folder
        folders.setdefault(k, {"folder": k, "label": v.label, "count": 0, "reason": v.reason})
        folders[k]["count"] += 1
    samples = []
    for v in scan.videos[:6]:
        cap = cv2.VideoCapture(str(v.path)); ok, frame = cap.read(); cap.release()
        thumb = ""
        if ok:
            _, buf = cv2.imencode('.jpg', cv2.resize(frame, (160, 90)))
            import base64
            thumb = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        samples.append({"video": v.path.name, "label": v.label, "thumb": thumb})
    return {
        "dataset_name": root.name,
        "total_videos": len(scan.videos),
        "normal_videos": sum(1 for v in scan.videos if v.label == 0),
        "anomaly_videos": sum(1 for v in scan.videos if v.label == 1),
        "unlabeled": 0,
        "has_masks": scan.has_masks,
        "folder_map": list(folders.values()),
        "estimated_frames": len(scan.videos) * 120,
        "size_mb": round(sum(v.path.stat().st_size for v in scan.videos) / (1024 * 1024), 2) if scan.videos else 0,
        "sample_frames": samples,
        "warnings": ["Video thiếu frame-level label sẽ dùng weak label."] if scan.videos else ["Không có video hợp lệ."],
    }


app.post("/api/training/extract")(make_sse_endpoint(trainer.run_extraction))
app.post("/api/training/train")(make_sse_endpoint(trainer.run_training))


@app.post("/api/training/evaluate")
async def training_eval(payload: dict):
    return trainer.evaluate(payload["session_id"])


@app.get("/api/training/sessions")
async def training_sessions():
    reg = DATA_DIR / "models" / "registry.json"
    return json.loads(reg.read_text(encoding="utf-8")).get("sessions", []) if reg.exists() else []


@app.get("/api/training/registry")
async def training_registry():
    reg = DATA_DIR / "models" / "registry.json"
    return json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {"sessions": [], "active_session": None}


@app.post("/api/training/activate")
async def training_activate(payload: dict):
    sid = payload["session_id"]
    reg_path = DATA_DIR / "models" / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {"sessions": [], "active_session": None}
    for s in reg.get("sessions", []):
        s["active"] = s["session_id"] == sid
    reg["active_session"] = sid
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    processor._load_active_model()
    return {"ok": True, "active_session": sid}


@app.delete("/api/training/session/{session_id}")
async def training_delete(session_id: str):
    reg_path = DATA_DIR / "models" / "registry.json"
    if not reg_path.exists():
        raise HTTPException(404, "Không có registry")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    new_s = []
    for s in reg.get("sessions", []):
        if s["session_id"] == session_id:
            for fp in s.get("files", {}).values():
                Path(fp).unlink(missing_ok=True)
        else:
            new_s.append(s)
    reg["sessions"] = new_s
    if reg.get("active_session") == session_id:
        reg["active_session"] = None
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    processor._load_active_model()
    return {"ok": True}
