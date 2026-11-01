"""
OCR 引擎和图像预处理
=====================
包含：印章图像预处理（红通道增强、二值化、极坐标展开等）、OCR 引擎封装
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Windows GPU：将 nvidia CUDA DLL 路径加入 PATH，避免 cublasLt64_13.dll 找不到
_nvidia_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ocr_venv", "Lib", "site-packages", "nvidia", "cu13", "bin", "x86_64")
if os.path.isdir(_nvidia_bin):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

# PaddleX 模型缓存路径（避免放 C 盘，且路径不能含中文）
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "D:\\paddlex_cache")

# ──────────────────────────────────────────────
# 1. 印章图像预处理
# ──────────────────────────────────────────────


def extract_red_channel(img: np.ndarray, alpha: float = 2.0, beta: float = -50) -> np.ndarray:
    """
    提取印章红色印文，增强红色通道，抑制背景。

    原理：中国公章多为朱红（R≫G,B），利用颜色分离增强红色区域。

    Args:
        img: BGR 图像 (H, W, 3)
        alpha: 红色通道增益系数
        beta: 绿蓝通道抑制偏移

    Returns:
        灰度图 (H, W)，红色区域亮、背景暗
    """
    if img is None or img.size == 0:
        raise ValueError("输入图像为空")

    b, g, r = cv2.split(img)
    r_enhanced = cv2.convertScaleAbs(r, alpha=alpha, beta=0)
    g_suppressed = cv2.convertScaleAbs(g, alpha=0.5, beta=beta)
    b_suppressed = cv2.convertScaleAbs(b, alpha=0.5, beta=beta)

    red_response = cv2.subtract(r_enhanced, cv2.addWeighted(g_suppressed, 0.5, b_suppressed, 0.5, 0))
    red_response = cv2.normalize(red_response, None, 0, 255, cv2.NORM_MINMAX)

    return red_response


def binarize_seal(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    对印章灰度图进行二值化。

    Args:
        gray: 灰度图
        method: "adaptive" | "otsu" | "triangle"

    Returns:
        二值图 (H, W)，文字为白色（255），背景为黑色（0）
    """
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=5
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "triangle":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        raise ValueError(f"未知的二值化方法: {method}")

    return binary


def denoise_seal(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    形态学去噪，去除孤立噪声点，连接断开的文字笔画。

    Args:
        binary: 二值图
        kernel_size: 形态学核大小

    Returns:
        去噪后的二值图
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def detect_seal_color(img: np.ndarray) -> str:
    """
    仅检测红色印章，非红色统一返回 "other"。

    判断依据：HSV 中红色像素（H=0~15° 或 160~180°，S>20）占比 >1.5%
    """
    h, w = img.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色（HSV 色环两端）
    red1 = cv2.inRange(hsv, (0, 20, 20), (15, 255, 255))
    red2 = cv2.inRange(hsv, (160, 20, 20), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = cv2.countNonZero(red_mask) / total

    return "red" if red_ratio > 0.015 else "other"


def preprocess_red_seal(
    img: np.ndarray,
    alpha: float = 2.0,
    beta: int = -50,
    binarize_method: str = "adaptive",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    红色印章预处理：红通道增强 → 二值化 → 形态学去噪 → 反转为白底黑字。
    """
    red = extract_red_channel(img, alpha=alpha, beta=beta)
    binary = binarize_seal(red, method=binarize_method)
    cleaned = denoise_seal(binary, kernel_size=kernel_size)
    return cv2.bitwise_not(cleaned)


def extract_red_channel_v2(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(
        hsv,
        (0, 20, 20),
        (15, 255, 255)
    )

    red2 = cv2.inRange(
        hsv,
        (160, 20, 20),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(red1, red2)

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    _, a, _ = cv2.split(lab)

    a_norm = cv2.normalize(
        a,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    result = cv2.addWeighted(
        a_norm,
        0.65,
        red_mask,
        0.35,
        0
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    result = clahe.apply(result)

    return result


def preprocess_red_seal_v2(img):
    """改进版红色印章预处理，减少笔画粘连，保护边缘文字。"""
    gray = extract_red_channel_v2(img)

    # 使用更大的 blockSize 适应印章文字粗细
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,   # 增大 blockSize，减少局部噪声
        7     # 增大 C，避免过度二值化
    )

    # 使用更小的核，避免侵蚀细笔画
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return cv2.bitwise_not(binary)


def preprocess_black_seal(
    img: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    黑色印章预处理：灰度化 → CLAHE 增强 → 自适应二值化 → 形态学去噪 → 反转为白底黑字。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE 自适应直方图均衡化，提升低对比度文字
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=5,
    )
    cleaned = denoise_seal(binary, kernel_size=kernel_size)
    return cv2.bitwise_not(cleaned)


def preprocess_other_seal(img: np.ndarray) -> np.ndarray:
    """
    其他颜色印章：转灰度 → 自适应二值化 → 反转为白底黑字。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=7,
    )
    return cv2.bitwise_not(binary)


def preprocess_blue_seal(
    img: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    蓝色印章预处理：提取蓝色通道 → CLAHE 增强 → 二值化 → 去噪 → 反转为白底黑字。
    """
    # 提取蓝色通道
    b, g, r = cv2.split(img)
    # 蓝色减去红绿（突出蓝色文字）
    blue_enhanced = cv2.subtract(b, cv2.addWeighted(g, 0.3, r, 0.3, 0))
    blue_enhanced = cv2.normalize(blue_enhanced, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blue_enhanced)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=5,
    )
    cleaned = denoise_seal(binary, kernel_size=kernel_size)
    return cv2.bitwise_not(cleaned)


def pad_for_seal(img: np.ndarray, pad_ratio: float = 0.15) -> np.ndarray:
    """
    为印章图像添加白边，防止边缘文字在后续处理中被裁剪。

    Args:
        img: 输入图像
        pad_ratio: 边距占原图尺寸的比例

    Returns:
        加边后的图像
    """
    h, w = img.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)

    if len(img.shape) == 3:
        # 彩色图像，填充白色 (255, 255, 255)
        padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        # 灰度/二值图像，填充白色 255
        padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                    cv2.BORDER_CONSTANT, value=255)
    return padded


def detect_seal_region(img: np.ndarray) -> np.ndarray | None:
    """
    检测印章区域并裁剪，多线索融合：HSV红色检测 + 圆形检测 + 轮廓面积。

    流程：
    1. 多线索检测印章位置
    2. 取并集区域
    3. 裁剪到该区域的包围矩形

    如果所有线索都失败则返回 None（使用原图）。
    """
    h, w = img.shape[:2]
    color = detect_seal_color(img)

    # 线索1：HSV 红色掩膜
    mask_hsv = None
    if color == "red":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 30, 30), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 30, 30), (180, 255, 255))
        mask_hsv = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)

    # 线索2：圆形检测
    circle = detect_seal_circle(img)
    mask_circle = None
    if circle is not None:
        cx, cy, r = circle
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        # 放大一点确保包含整个印章
        cv2.circle(mask_circle, (cx, cy), int(r * 1.2), 255, -1)

    # 线索3：轮廓面积（Canny 边缘）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_contour = np.zeros((h, w), dtype=np.uint8)
    if contours:
        big_contours = [c for c in contours if cv2.contourArea(c) > 500]
        cv2.drawContours(mask_contour, big_contours, -1, 255, -1)

    # 融合：任一线索有结果就取最大连通区域
    masks = [m for m in [mask_hsv, mask_circle, mask_contour] if m is not None]
    if not masks:
        return None

    # 融合掩膜
    fused = masks[0]
    for m in masks[1:]:
        fused = cv2.bitwise_or(fused, m)

    # 形态学闭运算连通
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)

    # 找最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused, connectivity=8)
    if num_labels < 2:
        return None

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    best_label = areas[0][0]

    x = stats[best_label, cv2.CC_STAT_LEFT]
    y = stats[best_label, cv2.CC_STAT_TOP]
    cw = stats[best_label, cv2.CC_STAT_WIDTH]
    ch = stats[best_label, cv2.CC_STAT_HEIGHT]

    if cw < 10 or ch < 10:
        return None

    # 扩大 20% 确保完整
    pad_x = int(cw * 0.20)
    pad_y = int(ch * 0.20)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    cropped = img[y1:y2, x1:x2]
    return cropped


def preprocess_seal_image(
    img: np.ndarray,
    alpha: float = 2.0,
    beta: int = -50,
    binarize_method: str = "adaptive",
    kernel_size: int = 3,
    invert: bool = True,
) -> np.ndarray:
    """
    印章图像预处理，返回二值图（兼容旧接口）。
    """
    versions = preprocess_multi_version(img, alpha, beta, binarize_method, kernel_size)
    result = versions.get("binary", img)
    if not invert and len(result.shape) == 2:
        result = cv2.bitwise_not(result)
    return result


def preprocess_multi_version(
    img: np.ndarray,
    alpha: float = 2.0,
    beta: int = -50,
    binarize_method: str = "adaptive",
    kernel_size: int = 3,
) -> dict:
    """
    对印章图像生成多个预处理版本，供多策略OCR使用。

    返回:
        {
            "gray":     灰度图
            "clahe":    CLAHE增强灰度图
            "red":      红色通道增强图（红色印章专用）
            "binary":   二值图（白底黑字）
            "color":    原始彩色图（加白边）
        }
    """
    img = pad_for_seal(img, pad_ratio=0.12)
    h, w = img.shape[:2]
    color = detect_seal_color(img)
    versions = {}

    # 1. 原始彩色图
    versions["color"] = img

    # 2. 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    versions["gray"] = gray

    # 3. CLAHE 增强灰度图（不二值化，保留细节）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe2.apply(enhanced)
    versions["clahe"] = enhanced

    # 4. 红色通道增强
    if color == "red":
        red_v1 = extract_red_channel(img, alpha=alpha, beta=beta)
        red_v2 = extract_red_channel_v2(img)
        versions["red"] = red_v2  # v2 边缘保护更好
    else:
        versions["red"] = enhanced

    # 5. 二值图（白底黑字，作为最后备选）
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=7,
    )
    cleaned = denoise_seal(binary, kernel_size=kernel_size)
    versions["binary"] = cv2.bitwise_not(cleaned)

    return versions


# ──────────────────────────────────────────────
# 2. 圆形印章极坐标展开
# ──────────────────────────────────────────────


def detect_seal_circle(img: np.ndarray) -> tuple | None:
    """
    检测印章的圆形区域。

    策略：
    1. 霍夫圆变换（快速准确）
    2. 边缘检测 + 轮廓拟合圆（霍夫失败的 fallback）

    Returns:
        (cx, cy, r) 圆心坐标和半径（Python int），未检测到时返回 None
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.medianBlur(gray, 5)

    # 策略1：霍夫圆变换
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=30,
        minRadius=20,
        maxRadius=min(h, w) // 2,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        largest = circles[circles[:, 2].argmax()]
        cx, cy, r = int(largest[0]), int(largest[1]), int(largest[2])
        if r > 10:
            return (cx, cy, r)

    # 策略2：轮廓拟合圆（霍夫失败的 fallback）
    # 用 Canny 找边缘，然后找最大轮廓拟合圆
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 按面积排序，取前5个大轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # 太小跳过
                continue
            # 最小外接圆
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)
            # 验证：圆度（轮廓面积与圆的面积比）
            circle_area = np.pi * r * r
            if circle_area > 0:
                roundness = area / circle_area
                # 0.3~1.3 是合理的圆度范围（印章近似圆形）
                if 0.25 < roundness < 1.35 and r > 15 and r < min(h, w) // 2 + 10:
                    return (cx, cy, r)

    return None


def polar_unwrap_seal(
    img: np.ndarray,
    center: tuple | None = None,
    radius: int | None = None,
    out_width: int = 800,
    out_height: int = 200,
) -> np.ndarray:
    """
    将圆形印章的弧形文字通过极坐标展开为直线排列。改进版：保护外圈边缘文字。

    Args:
        img: 预处理后的二值图
        center: 圆心 (cx, cy)，None 则自动检测
        radius: 展开半径，None 则用检测半径的 90%（原 85%，保护外圈）
        out_width: 输出图像宽度
        out_height: 输出图像高度

    Returns:
        展开后的图像
    """
    h, w = img.shape[:2]

    if center is None:
        circle = detect_seal_circle(img)
        if circle is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy, detected_r = circle
            radius = radius or int(detected_r * 0.90)  # 0.85 → 0.90
    else:
        cx, cy = center
        radius = radius or min(h, w) // 3

    polar = cv2.warpPolar(
        img,
        (out_width, out_height),
        (cx, cy),
        radius,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    return polar


def auto_polar_unwrap(img: np.ndarray) -> np.ndarray | None:
    """
    自动检测圆形印章并展开，若非圆形则返回 None。

    增强版：尝试多半径展开，选择文字最清晰的。
    """
    circle = detect_seal_circle(img)
    if circle is None:
        return None

    cx, cy, r = circle

    # 尝试多个展开半径，覆盖不同位置的环形文字
    radius_ratios = [0.92, 0.88, 0.82, 0.75]
    best_result = None
    best_quality = 0

    for ratio in radius_ratios:
        text_radius = int(r * ratio)
        try:
            unwrapped = polar_unwrap_seal(img, center=(cx, cy), radius=text_radius)
            if unwrapped is None:
                continue
            quality = cv2.countNonZero(unwrapped) / (unwrapped.shape[0] * unwrapped.shape[1])
            if quality > best_quality:
                best_quality = quality
                best_result = unwrapped
        except Exception:
            continue

    if best_result is not None:
        logger.info(f"检测到圆形: 中心=({cx},{cy}), 半径={r}, 展开质量={best_quality:.3f}")

    return best_result


def polar_unwrap_ring(
    img: np.ndarray,
    ring: str = "outer",
    circle: tuple | None = None,
    out_width: int = 800,
    out_height: int = 200,
) -> np.ndarray | None:
    """
    对圆形印章的指定环形区域进行极坐标展开。

    公章结构：
      外环 (0.7R~1.0R) → 单位名称（如"中国共产主义青年团福建农林大学委员会"）
      内环 (0.35R~0.7R) → 其他文字

    Args:
        img: 预处理后的二值图
        ring: "outer" 外环 | "inner" 内环
        circle: (cx, cy, r) 预检测的圆形参数，None 则自动检测
        out_width, out_height: 展开输出尺寸

    Returns:
        展开后的图像，失败返回 None
    """
    if circle is None:
        circle = detect_seal_circle(img)
    if circle is None:
        return None

    cx, cy, r = circle
    h, w = img.shape[:2]

    # 创建环形掩膜
    mask = np.zeros((h, w), dtype=np.uint8)

    if ring == "outer":
        # 外环：0.65R ~ 1.0R（单位名称区域）
        inner_r = int(r * 0.65)
        outer_r = int(r * 1.0)
    else:
        # 内环：0.30R ~ 0.65R
        inner_r = int(r * 0.30)
        outer_r = int(r * 0.65)

    cv2.circle(mask, (cx, cy), outer_r, 255, -1)
    cv2.circle(mask, (cx, cy), inner_r, 0, -1)

    # 提取环区域
    masked = cv2.bitwise_and(img, img, mask=mask)

    # 在环区域上做极坐标展开
    ring_radius = outer_r
    polar = cv2.warpPolar(
        masked,
        (out_width, out_height),
        (cx, cy),
        ring_radius,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    return polar


def sort_circular_text(results: list[dict], circle_center: tuple | None = None) -> list[dict]:
    """
    将圆形印章的 OCR 识别结果按角度排序（从顶部开始顺时针）。

    原理：圆章文字围绕圆心排列，按角度（atan2）排序可以恢复正确阅读顺序。
    适用于圆形印章的"弧形"文字排列。

    Args:
        results: OCR 识别结果列表，每项含 text, confidence, bbox
        circle_center: (cx, cy) 圆心坐标，None 时使用图片中心

    Returns:
        按角度排序后的列表
    """
    if not results:
        return []

    # 从 bbox 计算所有点的中心，估算圆心
    if circle_center is None:
        all_pts = []
        for r in results:
            bbox = r.get("bbox", [])
            if bbox and len(bbox) >= 4:
                for pt in bbox:
                    if len(pt) == 2:
                        all_pts.append(pt)
        if all_pts:
            all_pts = np.array(all_pts)
            cx, cy = float(np.median(all_pts[:, 0])), float(np.median(all_pts[:, 1]))
        else:
            cx, cy = 0.5, 0.5  # fallback
    else:
        cx, cy = circle_center

    # 为每条结果计算角度（弧度 → 度，0~360）
    for r in results:
        bbox = r.get("bbox", [])
        if bbox and len(bbox) >= 4:
            # bbox 中心
            bx = sum(pt[0] for pt in bbox[:4]) / 4
            by = sum(pt[1] for pt in bbox[:4]) / 4
        else:
            bx, by = cx, cy

        # atan2 返回 -π 到 π，角度从右侧开始逆时针
        # 转换为 0~360，从正上方开始顺时针
        angle_rad = np.arctan2(by - cy, bx - cx)
        angle_deg = np.degrees(angle_rad)
        # 调整：0°=正右 → 0°=正上，方向逆时针→顺时针
        # 公式: (-angle + 90) % 360
        angle_sorted = (-angle_deg + 90) % 360
        r["_sort_angle"] = angle_sorted

    # 按角度排序
    sorted_results = sorted(results, key=lambda x: x.get("_sort_angle", 0))

    # 清理临时字段
    for r in sorted_results:
        r.pop("_sort_angle", None)

    return sorted_results


def upscale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    对图像进行上采样放大，提高小尺寸印章的 OCR 识别率。

    Args:
        img: 输入图像
        scale: 放大倍数

    Returns:
        放大的图像
    """
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def create_color_mask(img: np.ndarray, color: str) -> np.ndarray:
    """
    创建指定颜色的掩膜，只保留该颜色区域。

    Args:
        img: BGR 图像
        color: "red" | "black"

    Returns:
        掩膜图像，目标颜色区域为白色(255)，其他区域为黑色(0)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "red":
        # 红色在 HSV 色环两端（0~10° 和 170~180°）
        mask1 = cv2.inRange(hsv, (0, 30, 30), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 30, 30), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "black":
        # 黑色：低亮度 + 低饱和度（避免彩色黑色）
        # V < 80 且 S < 50（排除深色但有颜色的区域）
        v_mask = cv2.inRange(hsv[:, :, 2], 0, 80)
        s_mask = cv2.inRange(hsv[:, :, 1], 0, 50)
        mask = cv2.bitwise_and(v_mask, s_mask)
    else:
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

    # 膨胀掩膜，确保文字边缘被包含
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def apply_mask_to_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    将掩膜应用到图像上，只保留掩膜区域。

    Args:
        img: 原始图像（BGR 或灰度）
        mask: 掩膜图像

    Returns:
        应用掩膜后的图像
    """
    if len(img.shape) == 3:
        # BGR 图像
        return cv2.bitwise_and(img, img, mask=mask)
    else:
        # 灰度图像
        return cv2.bitwise_and(img, img, mask=mask)


def extract_text_ring(img: np.ndarray, inner_ratio: float = 0.42, outer_ratio: float = 1.05) -> np.ndarray:
    """
    从圆形印章中提取环形文字区域（去除中间的五角星和空白区域）。
    改进版：增大外环比例保护边缘文字，减小内环比例去除更多中心干扰。

    Args:
        img: 预处理后的二值图
        inner_ratio: 内环半径比例（0.50 → 0.42）
        outer_ratio: 外环半径比例（0.95 → 1.05，允许超出边界）

    Returns:
        仅包含环形文字区域的图像
    """
    circle = detect_seal_circle(img)
    if circle is None:
        return img

    cx, cy, r = circle
    h, w = img.shape[:2]
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    inner_r = int(r * inner_ratio)
    outer_r = int(r * outer_ratio)

    cv2.circle(mask, (cx, cy), outer_r, 255, -1)
    cv2.circle(mask, (cx, cy), inner_r, 0, -1)

    result = cv2.bitwise_and(img, img, mask=mask)
    return result


# ──────────────────────────────────────────────
# 3. OCR 引擎封装
# ──────────────────────────────────────────────


class OCREngine:
    """
    OCR 引擎封装。

    使用方式：
        engine = OCREngine()
        results = engine.recognize(image)
    """

    def __init__(self, backend: str = "paddleocr_seal", lang: str = "ch"):
        """
        Args:
            backend: "paddleocr_seal" (默认) | "paddleocr"
            lang: 识别语言，"ch"（中文）
        """
        self.backend = backend
        self.lang = lang
        self._engine = None
        self._initialized = False
        self._engine_type = None

    def _init_paddleocr_seal(self):
        """初始化 PaddleOCR 印章专用流水线"""
        try:
            from paddlex import create_pipeline

            self._engine = create_pipeline(pipeline="seal_recognition")
            self._engine_type = "paddleocr_seal"
            logger.info("PaddleOCR 印章识别流水线初始化成功")
            return True
        except ImportError:
            logger.warning("PaddleX 未安装，尝试通用 PaddleOCR...")
            return False
        except Exception as e:
            logger.warning(f"PaddleOCR 印章流水线初始化失败: {e}")
            return False

    def _init_paddleocr(self):
        """初始化 PaddleOCR 3.x 通用引擎"""
        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._engine_type = "paddleocr"
            logger.info("PaddleOCR v6 medium 引擎初始化成功")
            return True
        except ImportError:
            logger.warning("PaddleOCR 未安装")
            return False
        except Exception as e:
            logger.warning(f"PaddleOCR 初始化失败: {e}")
            return False

    def initialize(self):
        """初始化 OCR 引擎"""
        if self._initialized:
            return True

        if self.backend == "paddleocr_seal":
            if not self._init_paddleocr_seal():
                raise RuntimeError("PaddleOCR 印章流水线初始化失败")
        elif self.backend == "paddleocr":
            if not self._init_paddleocr():
                raise RuntimeError("PaddleOCR 初始化失败")
        else:
            raise ValueError(f"不支持的后端: {self.backend}")

        self._initialized = True
        return True

    def recognize(self, img: np.ndarray) -> list[dict]:
        """
        对图像进行文字识别。

        Returns:
            [{"text": str, "confidence": float, "bbox": [[x1,y1],...]}, ...]
        """
        self.initialize()

        if self._engine_type == "paddleocr_seal":
            return self._recognize_paddle_seal(img)
        else:
            return self._recognize_paddle(img)

    def _recognize_paddle_seal(self, img: np.ndarray) -> list[dict]:
        """PaddleOCR 印章专用流水线识别。保留圆形阅读顺序和位置信息。"""
        try:
            # seal_recognition 的 predict 需要 use_layout_detection=False
            # 因为输入已经是裁剪好的印章图，不需要版面检测
            result = self._engine.predict(img, use_layout_detection=False)
            parsed = []
            for res in result:
                j = res.json
                seal_list = j.get("res", {}).get("seal_res_list", [])
                for seal in seal_list:
                    if not isinstance(seal, dict):
                        continue
                    texts = seal.get("rec_texts", [])
                    scores = seal.get("rec_scores", [])
                    boxes = seal.get("rec_boxes", [])
                    # seal_recognition 的 rec_texts 已是圆形阅读顺序
                    # 外环→内环，顺时针排列，不要重排
                    for i, t in enumerate(texts):
                        if t and str(t).strip():
                            conf = float(scores[i]) if i < len(scores) else 0.0
                            box = boxes[i] if i < len(boxes) else []
                            parsed.append({
                                "text": str(t).strip(),
                                "confidence": conf,
                                "bbox": self._normalize_bbox(box),
                                "_seal_order": i,
                            })
            # 不按置信度排序：保持 seal_recognition 的圆形阅读顺序
            return parsed
        except Exception as e:
            logger.debug(f"印章专用流水线识别异常: {e}")
            return []

    def _parse_paddle_result(self, res) -> list[dict]:
        """解析单条 PaddleOCR 结果，兼容多种返回格式，返回 0~N 个识别结果。

        支持的格式：
        1. 对象属性: res.text / res.score / res.box（单个结果）
        2. dict: res["text"] / res["score"] / res["box"]（单个结果）
        3. 元组: (box, text, score)（单个结果）
        4. PaddleOCR 3.x predict: res.json → {rec_texts, rec_scores, rec_boxes}（多个结果）

        Returns:
            [{"text": str, "confidence": float, "bbox": list}, ...]
        """
        items = []

        # 格式4：PaddleOCR 3.x 的 json — 包含多段文本，优先处理
        json_data = None
        if hasattr(res, "json") and not callable(res.json):
            json_data = res.json
        elif hasattr(res, "json") and callable(res.json):
            # res.json() 方法返回字符串
            try:
                json_data = res.json()
                if isinstance(json_data, str):
                    import json as _json
                    json_data = _json.loads(json_data)
            except Exception:
                pass

        if isinstance(json_data, dict):
            res_data = json_data.get("res", json_data)
            texts = res_data.get("rec_texts", [])
            scores = res_data.get("rec_scores", [])
            boxes = res_data.get("rec_boxes", [])
            for i in range(len(texts)):
                t = texts[i]
                if not t or not str(t).strip():
                    continue
                s = float(scores[i]) if i < len(scores) else 0.0
                b = boxes[i] if i < len(boxes) else []
                b = self._normalize_bbox(b)
                items.append({"text": str(t).strip(), "confidence": s, "bbox": b})
            return items

        # 格式1：对象属性
        if hasattr(res, "text") and not callable(res.text):
            text = res.text
            score = getattr(res, "score", None)
            box = getattr(res, "box", None)
        elif isinstance(res, (list, tuple)):
            if len(res) == 3:
                # 格式3：元组 (box, text, score)
                box, text, score = res
            elif len(res) == 2:
                # 可能是 [text, score]
                text, score = res[0], res[1]
            else:
                text, score, box = None, None, None
        elif isinstance(res, dict):
            # 格式2：dict
            text = None
            for key in ("text", "txt", "rec_text", "words"):
                if key in res and res[key]:
                    text = res[key]
                    break
            score = None
            for key in ("score", "confidence", "scores", "rec_score"):
                if key in res and res[key] is not None:
                    score = res[key]
                    break
            box = None
            for key in ("box", "bbox", "boxes", "rec_box"):
                if key in res and res[key]:
                    box = res[key]
                    break
        else:
            text, score, box = None, None, None

        if not text or not str(text).strip():
            return []

        text = str(text).strip()
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
        else:
            score = 0.0

        box = self._normalize_bbox(box)
        return [{"text": text, "confidence": score, "bbox": box}]

    def _normalize_bbox(self, box) -> list:
        """统一 bbox 格式为 [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]"""
        if not box:
            return []
        if isinstance(box, (list, tuple)):
            if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                x1, y1, x2, y2 = map(int, box)
                return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            elif all(isinstance(pt, (list, tuple)) for pt in box):
                return box
        return []

    def _recognize_paddle(self, img: np.ndarray) -> list[dict]:
        """PaddleOCR 识别（兼容多版本返回格式）"""
        parsed = []
        try:
            results = self._engine.predict(img)
            for res in results:
                items = self._parse_paddle_result(res)
                if items:
                    parsed.extend(items)
        except Exception as e:
            logger.debug(f"PaddleOCR 识别异常: {e}")

        # 过滤无效结果
        parsed = [p for p in parsed if len(p.get("text", "")) >= 1]
        parsed.sort(key=lambda x: x["confidence"], reverse=True)
        return parsed
