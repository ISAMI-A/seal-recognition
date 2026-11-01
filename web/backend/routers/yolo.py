"""
YOLO 检测路由
"""
import os
import uuid
import json
import zipfile
import io
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from ..services.yolo_service import _run_yolo_detect, _run_yolo_detect_pdf

router = APIRouter(prefix="/yolo", tags=["YOLO"])

ROOT = Path(__file__).resolve().parents[3]  # 项目根目录 testAI/
UPLOAD_DIR = ROOT / "data" / "_web_uploads"
CROP_DIR = ROOT / "data" / "_web_crops"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传 PDF/图片用于 YOLO 检测"""
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "file")[1] or ""
        name = f"{uuid.uuid4().hex}{ext}"
        path = UPLOAD_DIR / name
        content = await f.read()
        path.write_bytes(content)
        saved.append({
            "id": name,
            "name": f.filename or name,
            "size": len(content),
            "ext": ext.lower(),
        })
    return {"files": saved}


@router.post("/detect")
async def detect(file_ids: str = Form(...)):
    """对上传的文件进行 YOLO 检测"""
    ids = json.loads(file_ids)
    if not ids:
        raise HTTPException(400, "请先上传文件")

    pdf_ids = []
    image_ids = []
    for fid in ids:
        p = UPLOAD_DIR / fid
        if not p.exists():
            continue
        if p.suffix.lower() in (".pdf",):
            pdf_ids.append(fid)
        else:
            image_ids.append(fid)

    if not pdf_ids and not image_ids:
        raise HTTPException(400, "没有有效的文件")

    all_crops = []
    task_ids = []
    errors = []

    if pdf_ids:
        batch_id = uuid.uuid4().hex
        pdf_dir = UPLOAD_DIR / f"_pdf_{batch_id}"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        try:
            for fid in pdf_ids:
                src = UPLOAD_DIR / fid
                dst = pdf_dir / src.name
                dst.write_bytes(src.read_bytes())

            task_id = uuid.uuid4().hex
            crop_sub = CROP_DIR / task_id
            result = _run_yolo_detect_pdf(str(pdf_dir), str(crop_sub))
            if result.get("success"):
                all_crops.extend(result["crops"])
                task_ids.append(task_id)
            else:
                errors.append(f"PDF 检测失败: {result.get('error', '未知错误')}")
        finally:
            shutil.rmtree(pdf_dir, ignore_errors=True)

    if image_ids:
        batch_id = uuid.uuid4().hex
        img_dir = UPLOAD_DIR / f"_img_{batch_id}"
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            for fid in image_ids:
                src = UPLOAD_DIR / fid
                dst = img_dir / src.name
                dst.write_bytes(src.read_bytes())

            task_id = uuid.uuid4().hex
            crop_sub = CROP_DIR / task_id
            result = _run_yolo_detect(str(img_dir), str(crop_sub))
            if result.get("success"):
                all_crops.extend(result["crops"])
                task_ids.append(task_id)
            else:
                errors.append(f"图片检测失败: {result.get('error', '未知错误')}")
        finally:
            shutil.rmtree(img_dir, ignore_errors=True)

    return {
        "success": len(errors) == 0 or len(all_crops) > 0,
        "crops": all_crops,
        "count": len(all_crops),
        "task_ids": task_ids,
        "errors": errors if errors else None,
    }


@router.delete("/files")
async def delete_files(file_ids: str = Form(...)):
    """批量删除上传的文件"""
    ids = json.loads(file_ids)
    deleted = 0
    for fid in ids:
        p = UPLOAD_DIR / fid
        if p.exists():
            p.unlink()
            deleted += 1
    return {"deleted": deleted}


@router.delete("/crops")
async def delete_crops(task_ids: str = Form(...)):
    """批量删除裁剪结果目录"""
    ids = json.loads(task_ids)
    deleted = 0
    for tid in ids:
        d = CROP_DIR / tid
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            deleted += 1
    return {"deleted": deleted}


@router.get("/crops/{task_id}/download")
def download_crops(task_id: str):
    """打包下载裁剪结果"""
    crop_dir = CROP_DIR / task_id
    if not crop_dir.exists():
        raise HTTPException(404, "裁剪结果不存在")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(crop_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=crops_{task_id}.zip"},
    )
