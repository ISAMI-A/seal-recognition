"""
OCR 识别路由
"""
import os
import uuid
import json
import csv
import io
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from ..services.ocr_service import get_ocr_service

router = APIRouter(prefix="/ocr", tags=["OCR"])

# 临时上传目录
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "_web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    """上传待识别的印章图片"""
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "image.jpg")[1] or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        path = UPLOAD_DIR / name
        content = await f.read()
        path.write_bytes(content)
        saved.append({
            "id": name,
            "name": f.filename or name,
            "size": len(content),
            "url": f"/static/uploads/{name}",
        })
    return {"files": saved}


@router.post("/recognize")
async def recognize(
    file_id: str = Form(...),
    mode: str = Form("general"),
):
    """对单张图片进行 OCR 识别"""
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(404, f"文件不存在: {file_id}")

    service = get_ocr_service()
    result = service.recognize(str(file_path), mode=mode)
    return {
        "file_id": file_id,
        "mode": mode,
        **result,
    }


@router.delete("/files")
async def delete_ocr_files(file_ids: str = Form(...)):
    """批量删除上传的 OCR 图片"""
    ids = json.loads(file_ids)
    deleted = 0
    for fid in ids:
        p = UPLOAD_DIR / fid
        if p.exists():
            p.unlink()
            deleted += 1
    return {"deleted": deleted}


@router.post("/export")
async def export_results(data: str = Form(...), fmt: str = Form("csv")):
    """导出识别结果"""
    records = json.loads(data)

    if fmt == "csv":
        output = io.StringIO()
        w = csv.writer(output)
        w.writerow(["文件名", "单位名称", "置信度", "全部文本", "识别方法"])
        for r in records:
            w.writerow([
                r.get("file_name", ""),
                r.get("org_name", ""),
                f"{r.get('confidence', 0):.2%}",
                " | ".join(r.get("all_texts", [])),
                r.get("method", ""),
            ])
        content = output.getvalue().encode("utf-8-sig")
        return StreamingResponse(
            io.BytesIO(content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=ocr_results.csv"},
        )

    elif fmt == "xlsx":
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "识别结果"
        ws.append(["文件名", "单位名称", "置信度", "全部文本", "识别方法"])
        for r in records:
            ws.append([
                r.get("file_name", ""),
                r.get("org_name", ""),
                f"{r.get('confidence', 0):.2%}",
                " | ".join(r.get("all_texts", [])),
                r.get("method", ""),
            ])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=ocr_results.xlsx"},
        )

    raise HTTPException(400, f"不支持的格式: {fmt}")
