"""通用印章识别流水线（独立，不依赖福州特定逻辑）"""
import cv2
import numpy as np
import logging
from pathlib import Path

from .engine import (
    OCREngine, upscale_image, logger,
    detect_seal_shape, warp_ellipse_to_circle,
)

from .extract import (
    clean_text, is_valid_org_name_general, merge_seal_texts,
)

logger = logging.getLogger(__name__)


def analyze_seal_general(img, ocr_engine=None):
    """
    通用印章分析入口。
    1. 形状检测 → 2. 仿射变换拉正 → 3. seal_recognition 识别
    → 4. 碎片合并 → 5. 清理/去符号 → 6. 输出
    """
    if ocr_engine is None:
        ocr_engine = OCREngine(backend="paddleocr_seal")
    
    # 1. 形状检测
    shape_info = detect_seal_shape(img)
    logger.info(f"印章形状: {shape_info['shape']}")
    
    # 2. 椭圆/矩形拉正
    warped = warp_ellipse_to_circle(img, shape_info)
    
    # 3. seal_recognition 识别
    try:
        ocr_engine.initialize()
        raw_results = ocr_engine.recognize(warped)
    except Exception as e:
        logger.warning(f"seal_recognition 失败: {e}")
        try:
            img_large = upscale_image(warped, scale=2.0)
            raw_results = ocr_engine.recognize(img_large)
        except Exception:
            raw_results = []
    
    # 4. 后处理
    all_texts = [r.get("text", "") for r in raw_results if r.get("text")]
    
    # 碎片合并
    sorted_results = sorted(raw_results, key=lambda x: x.get("_seal_order", 0) if "_seal_order" in x else 0)
    merged_candidates = merge_seal_texts(sorted_results)
    
    # 提取候选
    org_name = ""
    confidence = 0.0
    best_method = "none"
    
    for merged_text, merged_conf in merged_candidates:
        cleaned = clean_text(merged_text)
        if cleaned and len(cleaned) >= 4 and is_valid_org_name_general(cleaned):
            if merged_conf > confidence:
                org_name = cleaned
                confidence = merged_conf
                best_method = "seal_recognition"

    if not org_name:
        for r in raw_results:
            text = r.get("text", "").strip()
            conf = r.get("confidence", 0.0)
            cleaned = clean_text(text)
            if cleaned and len(cleaned) >= 4 and is_valid_org_name_general(cleaned):
                if conf > confidence:
                    org_name = cleaned
                    confidence = conf
                    best_method = "seal_recognition"
    
    # seal_recognition 未识别出，尝试放大图
    if not org_name:
        try:
            img_large = upscale_image(warped, scale=2.0)
            raw_large = ocr_engine.recognize(img_large)
            for r in raw_large:
                text = r.get("text", "").strip()
                conf = r.get("confidence", 0.0)
                cleaned = clean_text(text)
                if cleaned and len(cleaned) >= 4 and is_valid_org_name_general(cleaned):
                    if conf > confidence:
                        org_name = cleaned
                        confidence = conf * 0.9
                        best_method = "upscaled"
        except Exception:
            pass

    return {
        "org_name": org_name,
        "confidence": confidence,
        "method": best_method,
        "all_texts": all_texts,
        "all_strategies": [best_method],
        "candidates": [org_name] if org_name else [],
        "shape_info": shape_info,
        "warped": warped,
        "ocr_details": [
            {"text": r.get("text", ""), "confidence": r.get("confidence", 0.0), "bbox": r.get("bbox", [])}
            for r in raw_results if r.get("text")
        ],
    }


def analyze_seal_file_general(image_path, **kwargs):
    """从文件路径读取图片并调用 analyze_seal_general。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片读取失败")
    result = analyze_seal_general(img, **kwargs)
    result["image_file"] = str(path)
    return result
