"""
端到端印章分析流程
==================
多策略 OCR + 单位名称提取
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import re

from .engine import (
    OCREngine, preprocess_seal_image, preprocess_multi_version, upscale_image,
    extract_text_ring, auto_polar_unwrap, polar_unwrap_ring,
    detect_seal_color, detect_seal_circle, extract_red_channel, logger,
)

from .extract import (
    extract_org_name, is_org_name, is_likely_body_text,
    correct_similar_chars, match_known_organization,
    org_structure_score, levenshtein_auto_correct,
)


# 策略权重：内外环极坐标展开权重最高，原图权重最低
STRATEGY_WEIGHT = {
    "polar_outer": 1.8,
    "polar_inner": 1.5,
    "polar": 1.3,
    "upscaled": 1.3,       # 对比分析：正确率88%，提升权重
    "text_ring": 1.2,
    "color_red": 0.7,      # 对比分析：正确率仅53%，降低权重
    "color_blue": 0.7,
    "direct": 0.8,
}


def _add_result(result, strategy, strategy_results, all_texts, candidates):
    """将单个策略的识别结果添加到汇总集合中。改进版：不过早过滤，保留所有候选。"""
    if not result:
        return

    strategy_results[strategy] = result

    for t in result.get("all_texts", []):
        if t and len(t) >= 2:
            all_texts.add(t)

    name = result.get("org_name", "")
    if not name:
        return

    # 仍然过滤明显混乱的结果
    if re.search(r"\d+.*汉族|汉族.*\d+|共青.*\d{2,}", name):
        logger.debug(f"策略 {strategy} 结果混乱，过滤: {name}")
        return

    weight = STRATEGY_WEIGHT.get(strategy, 1.0)
    conf = result.get("confidence", 0)
    raw_score = min(conf * weight, 1.0)

    # 结构评分辅助判断
    struct_score = org_structure_score(name)

    # 检查是否已有相似候选，有则合并提升
    for c in candidates:
        if c["name"] in name or name in c["name"] or _text_similarity(c["name"], name) > 0.6:
            # 取更长的名字
            if len(name) > len(c["name"]):
                c["name"] = name
            c["score"] = max(c["score"], min(raw_score * 1.15, 1.0))
            c["strategies"].add(strategy)
            c["struct_score"] = max(c.get("struct_score", 0), struct_score)
            return

    candidates.append({
        "name": name,
        "confidence": conf,
        "score": raw_score,
        "struct_score": struct_score,
        "strategy": strategy,
        "strategies": {strategy},
    })


def _text_similarity(a, b):
    """计算两段文本的相似度（字符集合重叠率）。"""
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def _vote_for_best(candidates, min_agreement=1):
    """
    多策略投票选出最佳结果。

    规则：
    1. 多策略一致（>=2个策略识别出相似结果）→ 选名字最长的
    2. 单策略 → 按 结构评分 + 策略数 + 命名长度 综合排序
    """
    if not candidates:
        return None

    # 多策略一致优先
    multi = [c for c in candidates if len(c.get("strategies", set())) >= min_agreement]
    if multi:
        best = max(multi, key=lambda x: (len(x["strategies"]), len(x["name"]), x.get("struct_score", 0)))
        return best

    # 单策略：结构评分 * 策略权重
    for c in candidates:
        w = STRATEGY_WEIGHT.get(c["strategy"], 1.0)
        c["_rank"] = c.get("struct_score", 0) * w + len(c["name"]) * 0.5

    return max(candidates, key=lambda x: x["_rank"])


def _retry_with_fallback(img, ocr_engine, circle_center, img_hw, strategy_results, all_texts, candidates):
    """当主要策略都失败时，使用备用参数重试。"""
    logger.info("主要策略未识别出有效结果，尝试备用参数...")

    for label, enhanced in [
        ("fallback_contrast", _clahe_enhance(img)),
        ("fallback_small", _downscale(img)),
    ]:
        if enhanced is None:
            continue
        try:
            r = ocr_engine.recognize(enhanced)
            out = extract_org_name(r, circle_center, img_hw)
            _add_result(out, label, strategy_results, all_texts, candidates)
        except Exception as e:
            logger.debug(f"{label}失败:{e}")


def _clahe_enhance(img):
    """CLAHE 对比度增强。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _downscale(img):
    """降分辨率（当上采样过度时有用）。"""
    h, w = img.shape[:2]
    if h > 300 and w > 300:
        return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return None


def analyze_seal(img, ocr_engine=None, enable_polar_unwrap=True,
                 preprocessing_params=None, upscale_factor=2.0):

    if ocr_engine is None:
        ocr_engine = OCREngine(backend="paddleocr")

    # === 快速路径：seal_recognition 专用（跳过冗余的多策略） ===
    # seal_recognition 内部已处理：印章检测、圆形文字排列、文字识别
    # 多策略（极坐标展开、颜色增强等）对它而言是多余的
    if ocr_engine.backend == "paddleocr_seal":
        try:
            ocr_engine.initialize()
            raw_results = ocr_engine.recognize(img)
            out = extract_org_name(
                raw_results,
                circle_center=None,
                img_shape=img.shape[:2],
            )
            if out and out.get("org_name"):
                name = out["org_name"]
                conf = out["confidence"]
                # 质量检查：带数字前缀的结果回退多策略；置信度过低（<70%）也回退
                has_digit_prefix = bool(re.match(r"^\d+", name))
                low_conf = conf < 0.70
                if has_digit_prefix or low_conf:
                    reason = "带数字前缀" if has_digit_prefix else f"置信度{conf:.0%}<70%"
                    logger.info(f"seal_recognition 结果质量不足({reason}): {name}，回退多策略...")
                else:
                    logger.info(f"seal_recognition 快速路径: {name}")
                    # 生成预处理图供前端可视化
                    from .engine import preprocess_red_seal, detect_seal_color
                    seal_color = detect_seal_color(img)
                    if seal_color == "red":
                        fast_processed = preprocess_red_seal(img)
                    else:
                        fast_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        fast_processed = cv2.bitwise_not(cv2.adaptiveThreshold(
                            fast_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 41, 7))
                    return {
                        "org_name": name,
                        "confidence": conf,
                        "method": "seal_recognition",
                        "ocr_raw": out.get("all_texts", []),
                        "all_strategies": ["seal_recognition"],
                        "all_texts": out.get("all_texts", []),
                        "candidates": out.get("candidates", []),
                        "preprocessed": fast_processed,
                        "unwrapped": None,
                        "ocr_details": [
                            {"text": r.get("text", ""), "confidence": r.get("confidence", 0.0), "bbox": r.get("bbox", [])}
                            for r in raw_results if r.get("text")
                        ],
                    }
            logger.info("seal_recognition 未识别出有效单位名，回退多策略...")
        except Exception as e:
            logger.warning(f"seal_recognition 快速路径失败: {e}，回退多策略...")

    # ── 以下为原有多策略路径（通用 OCR 备用） ──
    img_large = upscale_image(img, scale=upscale_factor)
    params = preprocessing_params or {}

    # 生成多版本预处理图
    versions = preprocess_multi_version(img, **params)
    versions_large = preprocess_multi_version(img_large, **params)

    preprocessed = versions.get("binary", img)
    preprocessed_large = versions_large.get("binary", img_large)

    all_texts = set()
    strategy_results = {}
    candidates = []
    img_hw = img.shape[:2]

    # 检测圆章
    circle_center = None
    circle = None
    try:
        circle = detect_seal_circle(img)
        if circle:
            circle_center = (circle[0], circle[1])
    except:
        pass

    def add_result(result, strategy):
        _add_result(result, strategy, strategy_results, all_texts, candidates)

    def _ocr_and_add(img_input, strategy, center=None):
        """辅助：OCR 识别 + 提取 + 添加结果"""
        try:
            r = ocr_engine.recognize(img_input)
            out = extract_org_name(r, center or circle_center, img_hw)
            add_result(out, strategy)
        except Exception as e:
            logger.debug(f"{strategy}失败:{e}")

    # ── 策略 1: 原图 ──
    _ocr_and_add(img, "direct")

    # ── 策略 2: 放大 ──
    _ocr_and_add(img_large, "upscaled")

    # ── 策略 3: 环形文字 ──
    ring = extract_text_ring(preprocessed)
    if ring is not None:
        ring_img = cv2.cvtColor(ring, cv2.COLOR_GRAY2BGR)
        _ocr_and_add(ring_img, "text_ring")

    # ── 策略 4: 颜色增强 ──
    seal_color = None
    try:
        seal_color = detect_seal_color(img)
        if seal_color == "red":
            enhanced = extract_red_channel(img)
        elif seal_color == "blue":
            b, g, r_ch = cv2.split(img)
            enhanced = cv2.subtract(b, cv2.addWeighted(g, 0.3, r_ch, 0.3, 0))
        else:
            enhanced = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        _ocr_and_add(enhanced, f"color_{seal_color}", circle_center)
    except Exception as e:
        logger.debug(f"color失败:{e}")

    # ── 策略 5: 极坐标展开（全环） ──
    unwrapped = None
    if enable_polar_unwrap:
        try:
            unwrapped = auto_polar_unwrap(preprocessed_large)
            if unwrapped is not None:
                _ocr_and_add(unwrapped, "polar")
        except Exception as e:
            logger.debug(f"polar失败:{e}")

    # ── 策略 6: 外环极坐标展开（最高权重：单位名称区域） ──
    if enable_polar_unwrap and circle is not None:
        for ring_type in ("outer", "inner"):
            try:
                ring_unwrapped = polar_unwrap_ring(preprocessed_large, ring=ring_type, circle=circle)
                if ring_unwrapped is not None:
                    _ocr_and_add(ring_unwrapped, f"polar_{ring_type}")
            except Exception as e:
                logger.debug(f"polar_{ring_type}失败:{e}")

    # 如果所有策略都未产生候选，尝试备用
    if not candidates:
        _retry_with_fallback(img, ocr_engine, circle_center, img_hw,
                             strategy_results, all_texts, candidates)

    # ── 最终选择 ──
    best = _vote_for_best(candidates, min_agreement=1)

    if best:
        final_name = best["name"]

        # 尝试候选内取更长者
        for c in candidates:
            if final_name in c["name"] and len(c["name"]) > len(final_name):
                final_name = c["name"]
                best = c

        # 尝试编辑距离自动纠错
        corrected = levenshtein_auto_correct(final_name)
        if corrected and len(corrected) > len(final_name):
            logger.info(f"编辑距离纠错: {final_name} → {corrected}")
            final_name = corrected

        # 尝试已知单位匹配补全
        matched = match_known_organization(final_name)
        if matched and len(matched) > len(final_name):
            final_name = matched

        return {
            "org_name": final_name,
            "confidence": best["score"],
            "method": best["strategy"],
            "ocr_raw": strategy_results.get("direct", {}).get("all_texts", []),
            "all_strategies": list(strategy_results.keys()),
            "all_texts": list(all_texts),
            "candidates": [x["name"] for x in candidates],
            "preprocessed": preprocessed,
            "unwrapped": unwrapped,
            "ocr_details": [],
        }

    return {
        "org_name": "",
        "confidence": 0,
        "method": "none",
        "ocr_raw": strategy_results.get("direct", {}).get("all_texts", []),
        "all_strategies": list(strategy_results.keys()),
        "all_texts": list(all_texts),
        "candidates": [],
        "preprocessed": preprocessed,
        "unwrapped": unwrapped,
        "ocr_details": [],
    }


def analyze_seal_file(image_path, **kwargs):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片读取失败")

    result = analyze_seal(img, **kwargs)
    result["image_file"] = str(path)
    return result
