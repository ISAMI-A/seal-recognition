"""
印章 OCR 批量识别脚本
======================
从已有的裁剪印章图片中批量识别单位名称。

用法：
  python -m ocr.batch --input 截图 --output-json results.json
  python -m ocr.batch --pipeline --pdf data/pdf --output-dir ocr_output
"""

import json, sys, csv, argparse
from pathlib import Path

import cv2
import numpy as np
from ocr import OCREngine, analyze_seal_file, logger


def batch_analyze_seals(
    input_dir: str | Path,
    ocr_engine: OCREngine | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    enable_polar: bool = True,
    visualize: bool = False,
) -> list[dict]:
    """批量分析印章目录中的全部图片。"""
    input_path = Path(input_dir)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not files:
        logger.warning(f"未找到图片文件: {input_path}")
        return []

    logger.info(f"共 {len(files)} 张印章图片，开始 OCR 识别...")

    if ocr_engine is None:
        ocr_engine = OCREngine(backend="paddleocr")
        logger.info("正在初始化 OCR 引擎...")
        ocr_engine.initialize()

    results = []
    for i, f in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] 识别: {f.name}")

        try:
            result = analyze_seal_file(f, ocr_engine=ocr_engine, enable_polar_unwrap=enable_polar)
            result["file_name"] = f.name
            results.append(result)
            logger.info(f"  \u2713 {result['org_name'] or '(未识别)'} (conf={result['confidence']:.1%})")

            if visualize:
                _visualize_result(f, result)
        except Exception as e:
            logger.error(f"  \u2717 识别失败: {e}")
            results.append({"file_name": f.name, "org_name": "", "confidence": 0.0, "error": str(e)})

    recognized = [r for r in results if r.get("org_name")]
    logger.info(f"\n总数: {len(results)}, 识别出单位: {len(recognized)}, 识别率: {len(recognized)/len(results)*100:.1f}%")

    if output_json:
        _save_json(results, output_json)
    if output_csv:
        _save_csv(results, output_csv)

    return results


def pipeline_yolo_ocr(
    pdf_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    output_dir: str | Path = "ocr_output",
    output_json: str | Path | None = None,
    yolo_model: str = "yolo/models/印章识别模型.pt",
    enable_polar: bool = True,
) -> list[dict]:
    """完整管道：YOLO 检测印章 → 裁剪 → OCR 识别。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    crop_dir = output_path / "crops"
    crop_dir.mkdir(exist_ok=True)

    images = []

    if pdf_dir:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from yolo.tools.render_pdf import render_pdf_to_images

            pdf_path = Path(pdf_dir)
            render_out = output_path / "rendered"
            for pdf_file in pdf_path.glob("*.pdf"):
                render_pdf_to_images(str(pdf_file), str(render_out), dpi=200)
            images = sorted(render_out.glob("*.png"))
        except Exception as e:
            logger.error(f"PDF 渲染失败: {e}")

    if image_dir:
        img_path = Path(image_dir)
        images = sorted(img_path.glob("*.*"))
        images = [f for f in images if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not images:
        logger.error("没有找到待处理图片")
        return []

    logger.info(f"待处理图片: {len(images)} 张")
    logger.info("加载 YOLO 模型...")
    from ultralytics import YOLO
    model = YOLO(yolo_model)

    for i, img_file in enumerate(images, 1):
        logger.info(f"[{i}/{len(images)}] 检测: {img_file.name}")
        results = model.predict(str(img_file), conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
        img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if results.boxes is not None:
            for bi, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_name = f"{img_file.stem}_seal{bi+1}.jpg"
                cv2.imencode('.jpg', crop)[1].tofile(str(crop_dir / crop_name))
                logger.info(f"  裁剪印章保存: {crop_name}")

    logger.info("\n开始 OCR 识别...")
    ocr_results = batch_analyze_seals(crop_dir, output_json=output_json, enable_polar=enable_polar)
    return ocr_results


def _save_json(results: list[dict], output_path: str | Path):
    clean = [{k: v for k, v in r.items() if k not in ("preprocessed", "unwrapped", "ocr_raw")} for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_path}")


def _save_csv(results: list[dict], output_path: str | Path):
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "单位名称", "置信度", "全部文本", "识别方法"])
        for r in results:
            writer.writerow([
                r.get("file_name", ""), r.get("org_name", ""),
                f"{r.get('confidence', 0):.2%}",
                " | ".join(r.get("all_texts", [])), r.get("method", ""),
            ])
    logger.info(f"CSV 已保存: {output_path}")


def _visualize_result(image_path: Path, result: dict):
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        label = f"Org: {result['org_name'] or '(unknown)'}  ({result['confidence']:.0%})"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if "preprocessed" in result and result["preprocessed"] is not None:
            pp = result["preprocessed"]
            if len(pp.shape) == 2:
                pp = cv2.cvtColor(pp, cv2.COLOR_GRAY2BGR)
            img_display = np.hstack([img, pp])
        else:
            img_display = img
        h, w = img_display.shape[:2]
        if w > 1200:
            scale = 1200 / w
            img_display = cv2.resize(img_display, (1200, int(h * scale)))
        cv2.imshow("Seal OCR Result", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning(f"可视化失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="印章 OCR 批量识别工具")
    source = parser.add_argument_group("输入源")
    source.add_argument("--input", "-i", type=str, help="印章图片目录或单张图片路径")
    source.add_argument("--pipeline", action="store_true", help="完整管道：YOLO 检测 + OCR")
    source.add_argument("--pdf", type=str, help="PDF 文件目录（配合 --pipeline 使用）")
    source.add_argument("--image-dir", type=str, help="图片目录（配合 --pipeline 使用）")

    ocr_args = parser.add_argument_group("OCR 参数")
    ocr_args.add_argument("--backend", choices=["paddleocr", "paddleocr_seal"], default="paddleocr_seal")
    ocr_args.add_argument("--no-polar", action="store_true", help="禁用极坐标展开")

    output = parser.add_argument_group("输出参数")
    output.add_argument("--output-json", "-oj", type=str, default="seal_ocr_results.json")
    output.add_argument("--output-csv", "-oc", type=str, default="seal_ocr_results.csv")
    output.add_argument("--output-dir", "-o", type=str, default="ocr_output")

    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--yolo-model", type=str, default="yolo/models/印章识别模型.pt")

    args = parser.parse_args()

    if args.pipeline:
        results = pipeline_yolo_ocr(
            pdf_dir=args.pdf, image_dir=args.image_dir,
            output_dir=args.output_dir, output_json=args.output_json,
            yolo_model=args.yolo_model, enable_polar=not args.no_polar,
        )
    elif args.input:
        ocr_engine = OCREngine(backend=args.backend)
        results = batch_analyze_seals(
            args.input, ocr_engine=ocr_engine,
            output_json=args.output_json, output_csv=args.output_csv,
            enable_polar=not args.no_polar, visualize=args.visualize,
        )
    else:
        parser.print_help()
        return 1

    recognized = [r for r in results if r.get("org_name")]
    if results:
        print(f"\n共 {len(results)} 个印章，识别出 {len(recognized)} 个单位（{len(recognized)/max(len(results),1)*100:.1f}%）")
    return 0


if __name__ == "__main__":
    exit(main())
