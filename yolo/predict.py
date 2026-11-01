"""
批量预测脚本
- 用训练好的模型预测图片中的印章
- 输出 YOLO txt（训练用） + labelU JSON（人工校验用）
"""
import json
import uuid
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 配置
MODEL_PATH = r"yolo\models\印章识别模型.pt"
PREDICT_DIR = Path(r"data\images_raw")
OUTPUT_TXT_DIR = Path(r"data\predictions_txt")
OUTPUT_JSON = Path(r"data\predictions_labelu.json")
CONF_THRESHOLD = 0.25
CLASS_NAME = "seal"  # labelU 中的标签名


def get_image_size(image_path):
    """获取图片尺寸（支持中文路径）"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def main():
    print("=" * 60)
    print("批量预测 - 印章检测")
    print(f"模型: {MODEL_PATH}")
    print(f"图片目录: {PREDICT_DIR}")
    print("=" * 60)

    # 加载模型
    model = YOLO(MODEL_PATH)

    # 获取所有图片
    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        f for f in PREDICT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    ])
    print(f"\n共找到 {len(image_files)} 张图片")

    # 创建输出目录
    OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    # 逐张预测
    labelu_samples = []
    stats = {"has_seal": 0, "no_seal": 0}
    order_counter = 0

    for i, img_file in enumerate(image_files, 1):
        stem = img_file.stem
        print(f"\r[{i}/{len(image_files)}] {stem[:40]:40s}", end="")

        # YOLO 预测
        results = model.predict(
            str(img_file),
            conf=CONF_THRESHOLD,
            imgsz=640,
            device="cuda:0",
            verbose=False,
        )
        result = results[0]

        # 获取图片尺寸
        img_h, img_w = result.orig_shape[:2]
        if img_w == 0 or img_h == 0:
            img_w, img_h = get_image_size(str(img_file))

        # 提取检测框
        boxes = result.boxes
        yolo_lines = []
        rect_results = []

        if boxes is not None and len(boxes) > 0:
            stats["has_seal"] += 1
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # YOLO 归一化坐标
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # YOLO txt
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # labelU rectTool result（绝对像素坐标）
                rect_x = max(0, x1)
                rect_y = max(0, y1)
                rect_w = min(x2 - x1, img_w - rect_x)
                rect_h = min(y2 - y1, img_h - rect_y)

                order_counter += 1
                rect_results.append({
                    "id": str(uuid.uuid4())[:8],
                    "x": rect_x,
                    "y": rect_y,
                    "label": CLASS_NAME,
                    "width": rect_w,
                    "height": rect_h,
                    "order": idx + 1,
                })
        else:
            stats["no_seal"] += 1

        # 保存 YOLO txt
        txt_path = OUTPUT_TXT_DIR / f"{stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # 构建 labelU result JSON 字符串
        result_data = {
            "width": img_w,
            "height": img_h,
            "rotate": 0,
            "annotations": [{
                "toolName": "rectTool",
                "result": rect_results
            }]
        }

        # 构建 labelU 样本（id 为数字）
        sample = {
            "id": i,  # 用序号作为数字 id
            "result": json.dumps(result_data, ensure_ascii=False),
            "url": str(img_file.resolve()),
            "folder": str(PREDICT_DIR.resolve()),
            "fileName": img_file.name,
        }
        labelu_samples.append(sample)

    # 保存 labelU JSON（数组格式）
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labelu_samples, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print("预测完成！")
    print(f"  有印章: {stats['has_seal']} 张")
    print(f"  无印章: {stats['no_seal']} 张")
    print(f"\n  YOLO txt:     {OUTPUT_TXT_DIR.resolve()}/")
    print(f"  labelU JSON:  {OUTPUT_JSON.resolve()}")
    print(f"\nlabelU 导入步骤：")
    print(f"  1. 打开 labelU 项目")
    print(f"  2. 导入 → 预标注文件 → 选 {OUTPUT_JSON.name}")
    print(f"  3. 逐张检查：✅对的保留 ❌错的删 ➕漏的补")
    print(f"  4. 导出 YOLO 格式 → 交给我合并训练")


if __name__ == "__main__":
    main()
