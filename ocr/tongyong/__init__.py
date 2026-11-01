"""通用印章识别模块"""
from .engine import (
    OCREngine,
    extract_red_channel,
    detect_seal_color,
    detect_seal_circle,
    detect_seal_shape,
    warp_ellipse_to_circle,
    adaptive_color_enhance,
    upscale_image,
    logger,
)
from .pipeline import (
    analyze_seal_general,
    analyze_seal_file_general,
)
