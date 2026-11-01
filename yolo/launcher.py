"""
YOLO 印章检测启动器
==================
运行前先激活虚拟环境：
  .\venv\Scripts\Activate.ps1

然后修改下面的配置即可运行。
python yolo\launcher.py

MODE 可选：
  "pdf"     - 从 PDF 开始：渲染 → 检测 → 裁剪（一键完成）
  "detect"  - 从已有图片开始：直接检测并裁剪
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════
# 用户配置区 - 修改下面几行即可
# ═══════════════════════════════════════════
MODE = "pdf"                # "pdf" | "detect"
INPUT_DIR = "data/pdf"      # 输入目录（相对路径直接写，Windows绝对路径要用 r"" 前缀）
CROP_DIR = "data/testyolo"     # 裁剪印章输出目录
DPI = 300                   # PDF 渲染分辨率（仅 pdf 模式有效）
# 示例：
#   INPUT_DIR = "data/testyolo"              # 相对路径
#   INPUT_DIR = r"D:\xxx\testAI\data\pdf"   # 绝对路径（加 r 前缀防转义）
# ═══════════════════════════════════════════


def _detect(input_dir: str, output_dir: str):
    """YOLO 检测并裁剪印章"""
    from ultralytics import YOLO
    import cv2
    import numpy as np

    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"错误：输入目录不存在 {inp}")
        return

    model = YOLO("yolo/models/印章识别模型.pt")
    files = sorted(inp.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    print(f"共 {len(files)} 张图片，开始识别...")
    total_seals = 0
    for idx, f in enumerate(files, 1):
        print(f"\r[{idx}/{len(files)}] {f.stem[:35]:35s}", end="")

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
                cv2.imencode('.jpg', crop)[1].tofile(str(out / out_name))
                total_seals += 1

    print(f"\n\n完成！处理 {len(files)} 张，截取 {total_seals} 个印章")
    print(f"保存到: {out.resolve()}/")


def run_pdf(pdf_dir: str, crop_dir: str, dpi: int):
    """PDF → 渲染 → 检测 → 裁剪"""
    from yolo.tools.render_pdf import batch_render_pdf

    render_dir = Path(crop_dir) / "_pdf_rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PDF模式] 渲染 PDF: {pdf_dir} → {render_dir}")
    results = batch_render_pdf(pdf_dir, str(render_dir), dpi)
    total = sum(len(v) for v in results.values())
    print(f"渲染完成：{len(results)} 个 PDF，{total} 张图片\n")

    _detect(str(render_dir), crop_dir)


def run_detect(input_dir: str, crop_dir: str):
    """直接检测已有图片"""
    print(f"[检测模式] 输入: {input_dir}")
    _detect(input_dir, crop_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO 印章检测启动器")
    parser.add_argument("--mode", choices=["pdf", "detect"],
                        default=MODE, help="运行模式")
    parser.add_argument("--input", "-i", default=INPUT_DIR, help="输入目录（PDF或图片）")
    parser.add_argument("--crop", default=CROP_DIR, help="裁剪输出目录")
    parser.add_argument("--dpi", type=int, default=DPI, help="PDF 渲染分辨率")
    args = parser.parse_args()

    if args.mode == "pdf":
        run_pdf(args.input, args.crop, args.dpi)
    elif args.mode == "detect":
        run_detect(args.input, args.crop)
