"""
印章 OCR 测试脚本
==================
测试单张印章图片的 OCR 识别效果、对比预处理参数、评估极坐标展开效果。

用法：
  python -m ocr.test --image data/crops/xxx_seal1.jpg
  python -m ocr.test --image data/crops/xxx_seal1.jpg --debug
  python -m ocr.test --image data/crops/xxx_seal1.jpg --compare
  python -m ocr.test --dir data/crops --summary
"""

import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ocr import (
    OCREngine,
    preprocess_seal_image,
    extract_red_channel,
    binarize_seal,
    denoise_seal,
    detect_seal_circle,
    polar_unwrap_seal,
    analyze_seal_file,
    logger,
)


def test_preprocessing(image_path: Path) -> dict:
    """测试预处理各步骤效果。"""
    logger.info(f"测试预处理: {image_path.name}")
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    steps = {"original": img}

    red = extract_red_channel(img)
    steps["red_channel"] = red
    logger.info("  红通道提取: 完成")

    steps["binary_adaptive"] = binarize_seal(red, "adaptive")
    steps["binary_otsu"] = binarize_seal(red, "otsu")
    logger.info("  二值化: 完成")

    steps["denoised"] = denoise_seal(steps["binary_adaptive"], kernel_size=3)
    logger.info("  形态学去噪: 完成")

    steps["preprocessed"] = preprocess_seal_image(img)
    logger.info("  预处理流水线: 完成")

    circle = detect_seal_circle(steps["preprocessed"])
    steps["circle"] = circle
    if circle:
        cx, cy, r = circle
        steps["polar"] = polar_unwrap_seal(steps["preprocessed"], center=(cx, cy), radius=int(r * 0.8))
        logger.info(f"  圆形检测: 中心=({cx},{cy}), 半径={r}")
    else:
        steps["polar"] = None
        logger.info("  圆形检测: 未检测到圆形")

    return steps


def show_debug_windows(steps: dict, title: str = "Debug"):
    """在窗口中显示各预处理步骤的图像。"""
    for name, img in steps.items():
        if img is None:
            continue
        display = img.copy()
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        if name == "preprocessed" and steps.get("circle"):
            cx, cy, r = steps["circle"]
            cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
        cv2.imshow(f"{title} - {name}", display)
    logger.info("按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_directory(directory: Path) -> dict:
    """测试整个目录中所有印章图片，统计识别结果。"""
    files = sorted(directory.glob("*.*"))
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not files:
        logger.warning(f"目录中没有图片: {directory}")
        return {"total": 0, "recognized": 0, "results": []}

    logger.info(f"目录测试: {directory} ({len(files)} 张图片)")

    ocr_engine = OCREngine(backend="paddleocr")
    ocr_engine.initialize()

    results = []
    for f in files:
        try:
            t0 = time.time()
            r = analyze_seal_file(f, ocr_engine=ocr_engine)
            elapsed = time.time() - t0
            results.append({"file": f.name, "org_name": r["org_name"], "confidence": r["confidence"], "time": round(elapsed, 2)})
            status = "\u2713" if r["org_name"] else "\u2717"
            logger.info(f"  [{status}] {f.name:40s} → {r['org_name'] or '(未识别)':30s} ({r['confidence']:.0%})")
        except Exception as e:
            results.append({"file": f.name, "org_name": "", "confidence": 0, "time": 0, "error": str(e)})
            logger.error(f"  [E] {f.name}: {e}")

    recognized = sum(1 for r in results if r.get("org_name"))
    logger.info(f"\n  总计: {len(results)}, 识别: {recognized}, 识别率: {recognized/len(results)*100:.1f}%")
    return {"total": len(results), "recognized": recognized, "results": results}


def main():
    parser = argparse.ArgumentParser(description="印章 OCR 测试工具")
    parser.add_argument("--image", "-i", type=str, help="单张印章图片路径")
    parser.add_argument("--dir", "-d", type=str, help="印章图片目录路径")
    parser.add_argument("--debug", action="store_true", help="显示各预处理步骤的窗口")
    parser.add_argument("--summary", action="store_true", help="测试整个目录后输出汇总")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"文件不存在: {image_path}")
            return 1

        if args.debug:
            steps = test_preprocessing(image_path)
            show_debug_windows(steps)
        else:
            result = analyze_seal_file(image_path)
            print(f"\n文件:      {image_path.name}")
            print(f"单位名称:  {result['org_name'] or '(未识别到单位名称)'}")
            print(f"置信度:    {result['confidence']:.1%}")
            print(f"识别方法:  {result['method']}")
            print(f"全部文本:  {', '.join(result['all_texts'])}")
            if result["candidates"]:
                print(f"候选名称:  {', '.join(result['candidates'])}")

    elif args.dir:
        result = test_directory(Path(args.dir))
        if args.summary:
            print(f"\n总图片数: {result['total']}, 识别成功: {result['recognized']}, 识别率: {result['recognized']/max(result['total'],1)*100:.1f}%")
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    exit(main())
