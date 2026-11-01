"""
OCR 文字提取模块
==================
从印章裁剪图中识别文字，提取单位名称。

使用方式：
  from ocr import OCREngine, analyze_seal, analyze_seal_file

子模块：
  ocr.fuzhou    — 福州专区（红色圆形印章 + 福州专属匹配）
  ocr.tongyong  — 通用专区（多颜色/多形状 + 通用后处理）
"""

from .fuzhou.engine import (
    OCREngine,
    extract_red_channel,
    extract_red_channel_v2,
    binarize_seal,
    denoise_seal,
    detect_seal_color,
    preprocess_red_seal,
    preprocess_red_seal_v2,
    preprocess_black_seal,
    preprocess_blue_seal,
    preprocess_seal_image,
    preprocess_multi_version,
    detect_seal_circle,
    detect_seal_region,
    polar_unwrap_seal,
    auto_polar_unwrap,
    polar_unwrap_ring,
    upscale_image,
    extract_text_ring,
    sort_circular_text,
    create_color_mask,
    apply_mask_to_image,
    logger,
)

from .fuzhou.extract import (
    is_org_name,
    org_structure_score,
    levenshtein_auto_correct,
    merge_ocr_lines,
    extract_org_name,
    ORG_SUFFIX_PATTERNS,
    STOP_WORDS,
    REGION_PREFIXES,
)

from .fuzhou.pipeline import (
    analyze_seal,
    analyze_seal_file,
)

from . import fuzhou
from . import tongyong
