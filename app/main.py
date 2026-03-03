from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.behavior_model import load_behavior_model
from app.services.demo_inference import analyze_frame_pair, analyze_image_frame, analyze_video_file
from app.services.object_tracking import ObjectDetector, SimpleObjectTracker

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TMP_DIR = DATA_DIR / "tmp_uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

app = FastAPI(
    title="DCAD Demo - Vision-Based CCG",
    description="Interactive demo with upload and webcam analysis",
    version="0.3.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.state.prev_frames: dict[str, np.ndarray] = {}
app.state.trackers: dict[str, SimpleObjectTracker] = {}
app.state.detector = ObjectDetector()
app.state.behavior_model = load_behavior_model(DATA_DIR / "dcad_behavior_model.pt")


class WebcamPayload(BaseModel):
    client_id: str
    frame_data_url: str


def load_json(name: str):
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return frame


def _frame_from_data_url(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid data URL")
    _, b64 = data_url.split(",", 1)
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Cannot decode base64 frame") from exc
    return _decode_image_bytes(raw)


def _tracker_for_client(client_id: str) -> SimpleObjectTracker:
    tracker = app.state.trackers.get(client_id)
    if tracker is None:
        tracker = SimpleObjectTracker()
        app.state.trackers[client_id] = tracker
    return tracker


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    project_info = load_json("project_info.json")
    workflow = load_json("workflow.json")
    novelty = load_json("novelty.json")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "project": project_info,
            "workflow": workflow,
            "novelty": novelty,
        },
    )


@app.get("/api/workflow", response_class=JSONResponse)
async def workflow_api():
    return JSONResponse(load_json("workflow.json"))


@app.get("/api/novelty", response_class=JSONResponse)
async def novelty_api():
    return JSONResponse(load_json("novelty.json"))


@app.post("/api/analyze/upload-image", response_class=JSONResponse)
async def analyze_uploaded_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    raw = await file.read()
    frame = _decode_image_bytes(raw)
    result = analyze_image_frame(
        frame,
        detector=app.state.detector,
        tracker=SimpleObjectTracker(max_missed=0),
        behavior_model=app.state.behavior_model,
    )
    return JSONResponse(result.__dict__)


@app.post("/api/analyze/upload-video", response_class=JSONResponse)
async def analyze_uploaded_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing video file")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        result = analyze_video_file(
            temp_path,
            detector=app.state.detector,
            tracker=SimpleObjectTracker(),
            behavior_model=app.state.behavior_model,
            max_frames=150,
        )
        return JSONResponse(result.__dict__)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/api/analyze/webcam-frame", response_class=JSONResponse)
async def analyze_webcam_frame(payload: WebcamPayload):
    cur_frame = _frame_from_data_url(payload.frame_data_url)
    prev_frame = app.state.prev_frames.get(payload.client_id)

    tracker = _tracker_for_client(payload.client_id)
    app.state.prev_frames[payload.client_id] = cur_frame

    if prev_frame is None:
        result = analyze_image_frame(
            cur_frame,
            detector=app.state.detector,
            tracker=tracker,
            behavior_model=app.state.behavior_model,
        )
        result.source_type = "webcam"
        return JSONResponse(result.__dict__)

    result = analyze_frame_pair(
        prev_frame,
        cur_frame,
        detector=app.state.detector,
        tracker=tracker,
        behavior_model=app.state.behavior_model,
    )
    return JSONResponse(result.__dict__)
