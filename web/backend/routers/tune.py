"""
模型调优路由 — 标注编辑与导出
"""
import os
import json
import uuid
import hashlib
import zipfile
import io
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/tune", tags=["调优"])

TUNE_DIR = Path(__file__).resolve().parents[3] / "data" / "_web_tune"
TUNE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_tune_images(files: list[UploadFile] = File(...), session: str = Form(None)):
    """上传用于标注的图片。复用已有 session（保证同一次标注会话的文件都在同一目录）"""
    if session and (TUNE_DIR / session).exists():
        session_dir = TUNE_DIR / session
    else:
        session = uuid.uuid4().hex
        session_dir = TUNE_DIR / session
        session_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "image.jpg")[1] or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        path = session_dir / name
        content = await f.read()
        path.write_bytes(content)
        saved.append({
            "id": name,
            "name": f.filename or name,
            "path": str(path),
            "url": f"/static/tune/{session}/{name}",
        })
    # 增量更新原始文件名映射（id -> 原始名），导出预标注 JSON 时使用
    files_map = {}
    files_map_path = session_dir / "_files.json"
    if files_map_path.exists():
        files_map = json.loads(files_map_path.read_text(encoding="utf-8"))
    files_map.update({s["id"]: s["name"] for s in saved})
    files_map_path.write_text(json.dumps(files_map, ensure_ascii=False), encoding="utf-8")
    return {"session": session, "files": saved}


@router.delete("/files")
async def delete_tune_files(data: str = Form(...)):
    """批量删除调优上传的文件和标注，清理空 session 目录"""
    items = json.loads(data)
    deleted = 0
    # 按 session 分组，避免 session 不一致导致删不到文件
    by_session: dict[str, list[str]] = {}
    for item in items:
        session = item.get("session", "")
        file_id = item.get("file_id", "")
        if not session or not file_id:
            continue
        by_session.setdefault(session, []).append(file_id)

    for session, file_ids in by_session.items():
        sdir = TUNE_DIR / session
        if not sdir.exists():
            continue
        # 读取该 session 的标注，删除图片时同步删除对应标注
        ann_path = sdir / "_annotations.json"
        all_ann = {}
        if ann_path.exists():
            all_ann = json.loads(ann_path.read_text(encoding="utf-8"))
        # 同步清理原始文件名映射
        files_map_path = sdir / "_files.json"
        files_map = {}
        if files_map_path.exists():
            files_map = json.loads(files_map_path.read_text(encoding="utf-8"))
        for file_id in file_ids:
            p = sdir / file_id
            if p.exists():
                p.unlink()
                deleted += 1
            all_ann.pop(file_id, None)
            files_map.pop(file_id, None)
        if all_ann:
            ann_path.write_text(json.dumps(all_ann, ensure_ascii=False, indent=2), encoding="utf-8")
        elif ann_path.exists():
            ann_path.unlink()
        if files_map:
            files_map_path.write_text(json.dumps(files_map, ensure_ascii=False), encoding="utf-8")
        elif files_map_path.exists():
            files_map_path.unlink()
        # 清理空 session 目录
        if sdir.exists() and not any(sdir.iterdir()):
            sdir.rmdir()
    return {"deleted": deleted}


@router.post("/import-annotations")
async def import_annotations(
    session: str = Form(...),
    data: str = Form(...),
):
    """导入已有标注数据（JSON）"""
    ann_path = TUNE_DIR / session / "_annotations.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(data, encoding="utf-8")
    return {"status": "imported"}


@router.get("/annotations/{session}/{file_id}")
def get_annotations(session: str, file_id: str):
    """获取单张图片的标注"""
    ann_path = TUNE_DIR / session / "_annotations.json"
    if ann_path.exists():
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        return data.get(file_id, {"boxes": []})
    return {"boxes": []}


@router.put("/annotations/{session}/{file_id}")
async def save_annotations(session: str, file_id: str, data: str = Form(...)):
    """保存单张图片的标注"""
    ann_path = TUNE_DIR / session / "_annotations.json"
    ann_dir = ann_path.parent
    ann_dir.mkdir(parents=True, exist_ok=True)

    all_ann = {}
    if ann_path.exists():
        all_ann = json.loads(ann_path.read_text(encoding="utf-8"))
    all_ann[file_id] = json.loads(data)
    ann_path.write_text(json.dumps(all_ann, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "saved"}


@router.post("/export")
def export_annotations(session: str = Form(...), fmt: str = Form("yolo"), file_ids: str = Form(None)):
    """导出标注数据。file_ids 为可选的 JSON 数组，只导出指定文件"""
    ann_path = TUNE_DIR / session / "_annotations.json"
    if not ann_path.exists():
        raise HTTPException(404, "无标注数据，请先保存标注后再导出")

    all_ann = json.loads(ann_path.read_text(encoding="utf-8"))
    if file_ids:
        keep = set(json.loads(file_ids))
        all_ann = {k: v for k, v in all_ann.items() if k in keep}
    if not all_ann:
        raise HTTPException(404, "所选文件没有标注数据")
    session_dir = TUNE_DIR / session

    if fmt == "json":
        # LabelU 预标注 JSON 格式（兼容 LabelU 导出的标注文件，可直接作为预标注导入）
        files_map = {}
        files_map_path = session_dir / "_files.json"
        if files_map_path.exists():
            files_map = json.loads(files_map_path.read_text(encoding="utf-8"))

        items = []
        for idx, (file_id, ann) in enumerate(all_ann.items(), start=1):
            img_path = session_dir / file_id
            if not img_path.exists():
                continue
            w, h = _get_img_size(str(img_path))
            results = []
            for order, box in enumerate(ann.get("boxes", []), start=1):
                results.append({
                    "id": uuid.uuid4().hex,
                    "x": box["x"],
                    "y": box["y"],
                    "label": box.get("label") or "seal",
                    "width": box["width"],
                    "height": box["height"],
                    "order": order,
                })
            # result 必须为 JSON 字符串（LabelU 校验 result should be a string）
            inner = {
                "width": w,
                "height": h,
                "rotate": 0,
                "annotations": [
                    {"toolName": "rectTool", "result": results},
                ],
            }
            items.append({
                "id": idx,  # LabelU 顶层 id 要求 number
                "result": json.dumps(inner, ensure_ascii=False),
                "url": f"/static/tune/{session}/{file_id}",
                "folder": str(session_dir),
                "fileName": files_map.get(file_id, file_id),
            })

        content = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="annotations.json"'},
        )

    # YOLO 官方训练格式：images/train + images/val + labels/train + labels/val + data.yaml
    class_names = {}
    for ann in all_ann.values():
        for box in ann.get("boxes", []):
            cls_id = int(box.get("class_id", 0))
            class_names.setdefault(cls_id, box.get("label") or "seal")

    # 固定划分：按文件 id 的 md5 哈希分成 train(80%) / val(20%)，可复现
    def _split(name: str) -> str:
        return "train" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 < 8 else "val"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_id, ann in all_ann.items():
            img_path = session_dir / file_id
            if not img_path.exists():
                continue
            split = _split(file_id)
            h, w = _get_img_size(str(img_path))
            lines = []
            for box in ann.get("boxes", []):
                cls_id = int(box.get("class_id", 0))
                x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
                nx, ny, nw, nh = x / w, y / h, bw / w, bh / h
                lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            stem = Path(file_id).stem
            ext = Path(file_id).suffix or ".jpg"
            zf.write(img_path, f"images/{split}/{stem}{ext}")
            if lines:
                zf.writestr(f"labels/{split}/{stem}.txt", "\n".join(lines))

        yaml_lines = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
        ]
        yaml_lines += [f"  {cls_id}: {class_names[cls_id]}" for cls_id in sorted(class_names)]
        zf.writestr("data.yaml", "\n".join(yaml_lines))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=annotations_yolo.zip"},
    )


def _get_img_size(path: str) -> tuple:
    import cv2
    import numpy as np
    # cv2.imread 不支持中文路径，用 imdecode + np.fromfile 读取
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img.shape[1], img.shape[0]
    return 640, 640
