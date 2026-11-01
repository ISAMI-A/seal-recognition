"""
批量识别印章并截取保存
用法：python yolo/detect.py [输入目录] [输出目录]
  默认输入：data/pdf_rendered
  默认输出：data/crops
"""
import sys
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo/models/印章识别模型.pt")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/crops")
INPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/pdf_rendered")
OUT.mkdir(exist_ok=True)

if not INPUT_DIR.exists():
    print(f"错误：输入目录不存在 {INPUT_DIR}")
    sys.exit(1)

files = sorted(INPUT_DIR.glob("*.*"))
files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
total_files = len(files)

print(f"共 {total_files} 张图片，开始识别...")

total_seals = 0
for idx, f in enumerate(files, 1):
    print(f"\r[{idx}/{total_files}] {f.stem[:35]:35s}", end="")

    results = model.predict(str(f), conf=0.25, iou=0.5, imgsz=640, device="cuda:0", verbose=False)[0]
    img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue

    if results.boxes is not None:
        for bi, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_name = f"{f.stem}_seal{bi+1}.jpg"
            cv2.imencode('.jpg', crop)[1].tofile(str(OUT / out_name))
            total_seals += 1

print(f"\n\n完成！处理 {total_files} 张，截取 {total_seals} 个印章")
print(f"保存到: {OUT.resolve()}/")
