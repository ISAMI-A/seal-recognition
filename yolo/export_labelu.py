"""
整合所有图片 → 预测 → labelU JSON + 截图
"""
import json, uuid, shutil, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

# 图片来源
SOURCES = [
    Path("data/images_new"),
    Path("data/img/仓山区"),
    Path("data/img/43f19f8899d37008d17fd199e57ce34.jpg"),
    Path("data/raw/pdf2"),
]
UNIFIED_DIR = Path("data/all_images")
OUT_JSON = Path("data/labels/all_labelu.json")
OUT_CROP = Path("data/crops")

# 1. 整合图片
print("=== 整合图片 ===")
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
img_map = {}  # filename -> (src_path, unified_name)

for src in SOURCES:
    files = [src] if src.is_file() else sorted(src.glob("*.*"))
    for f in files:
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # 去重：相同文件名加前缀
        name = f.name
        if name in img_map:
            name = f"{f.parent.name}_{name}"
        dst = UNIFIED_DIR / name
        shutil.copy2(str(f), str(dst))
        img_map[name] = f

print(f"共 {len(img_map)} 张图片 → {UNIFIED_DIR}/")

# 2. 预测
print("\n=== 模型预测 ===")
model = YOLO("yolo/models/印章识别模型.pt")
files = sorted(UNIFIED_DIR.iterdir())
labelu_samples = []

OUT_CROP.mkdir(exist_ok=True)
total_seals = 0

for i, f in enumerate(files, 1):
    print(f"\r[{i}/{len(files)}] {f.stem[:35]:35s}", end="")
    
    results = model.predict(str(f), conf=0.25, iou=0.5, imgsz=640, device="cuda:0", verbose=False)[0]
    img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = results.orig_shape[:2]
    
    rects = []
    if results.boxes is not None:
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # labelU 框
            rects.append({
                "id": str(uuid.uuid4())[:8],
                "x": x1, "y": y1,
                "label": "seal",
                "width": x2 - x1,
                "height": y2 - y1,
                "order": idx + 1,
            })
            # 截图
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crop_name = f"{f.stem}_seal{idx+1}.jpg"
                cv2.imencode('.jpg', crop)[1].tofile(str(OUT_CROP / crop_name))
                total_seals += 1

    labelu_samples.append({
        "id": i,
        "result": json.dumps({
            "width": w, "height": h, "rotate": 0,
            "annotations": [{"toolName": "rectTool", "result": rects}]
        }),
        "url": str(f.resolve()),
        "folder": str(UNIFIED_DIR.resolve()),
        "fileName": f.name,
    })

# 3. 保存
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(labelu_samples, f, ensure_ascii=False, indent=2)

has_seal = sum(1 for s in labelu_samples if json.loads(s["result"])["annotations"][0]["result"])
print(f"\n\n=== 完成 ===")
print(f"图片总数: {len(labelu_samples)}")
print(f"有印章:   {has_seal}")
print(f"无印章:   {len(labelu_samples) - has_seal}")
print(f"截图数:   {total_seals}")
print(f"\n统一图片: {UNIFIED_DIR}/")
print(f"labelU:   {OUT_JSON}")
print(f"截图:     {OUT_CROP}/")
print(f"\n操作流程:")
print(f"1. labelU 导入 {OUT_JSON.name}，逐张校验")
print(f"2. 导出 YOLO 格式")
print(f"3. 把导出的文件夹路径给我 → 迭代训练")
