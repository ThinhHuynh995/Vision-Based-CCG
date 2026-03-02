from pathlib import Path
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="DCAD Demo - Phát hiện bất thường và phân tích dòng người",
    description="Nền tảng minh họa Vision-Based CCG với FastAPI",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


def load_json(name: str):
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


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
