"""
Web 后端入口 — FastAPI
运行方式（在 ocr_venv 下）：
  uvicorn web.backend.main:app --reload --port 8000
"""
import sys
import os
import shutil
from pathlib import Path

# 确保能导入项目根模块
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import ocr, yolo, tune

app = FastAPI(title="印章识别系统", version="0.1.0")

# CORS — 允许前端 dev server 跨域（可用环境变量 WEB_CORS_ORIGINS 覆盖，逗号分隔）
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_allow_origins = [o.strip() for o in os.getenv("WEB_CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录 — 让前端能访问上传和裁剪的图片
DATA_DIR = ROOT / "data"
CROPS_DIR = DATA_DIR / "_web_crops"
UPLOADS_DIR = DATA_DIR / "_web_uploads"
TUNE_DIR = DATA_DIR / "_web_tune"

CROPS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TUNE_DIR.mkdir(parents=True, exist_ok=True)

# 挂载静态文件目录
app.mount("/static/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static/tune", StaticFiles(directory=str(TUNE_DIR)), name="tune")

# 注册路由
app.include_router(ocr.router, prefix="/api")
app.include_router(yolo.router, prefix="/api")
app.include_router(tune.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/data/clear")
def clear_web_data():
    """一键清空 Web 临时数据目录（上传/裁剪/模型调优），保留目录本身"""
    cleared = 0
    for d in (CROPS_DIR, UPLOADS_DIR, TUNE_DIR):
        for item in d.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            cleared += 1
    return {"cleared": cleared}
