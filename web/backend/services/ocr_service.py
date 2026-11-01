"""
OCR 识别服务 — 封装福州/通用两种模式
"""
import base64
import cv2
import numpy as np


def _img_to_base64(img: np.ndarray) -> str:
    """OpenCV 图片 → base64 data URI"""
    if img is None:
        return ""
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _make_red_on_white(img: np.ndarray) -> np.ndarray:
    """从原图提取红色通道，生成白底红字可视化图"""
    if img is None or img.size == 0:
        return img
    b, g, r = cv2.split(img)
    # 红色增强：r - max(g, b)
    red_enhanced = cv2.addWeighted(r.astype(np.float32), 2.0,
                                   cv2.max(g.astype(np.float32), b.astype(np.float32)), -1, 0)
    red_enhanced = np.clip(red_enhanced, 0, 255).astype(np.uint8)
    # 阈值分割提取红色区域
    _, red_mask = cv2.threshold(red_enhanced, 30, 255, cv2.THRESH_BINARY)
    # 生成白底红字图
    result = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)  # 白色背景
    result[red_mask > 0] = [0, 0, 200]  # 红色文字 (BGR)
    return result


def _draw_text_boxes(img: np.ndarray, details: list[dict]) -> np.ndarray:
    """在图像上绘制文字检测框"""
    if img is None or not details:
        return img
    vis = img.copy()
    for d in details:
        bbox = d.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        conf = d.get("confidence", 0)
        color = (0, int(255 * conf), int(255 * (1 - conf)))  # 高置信度绿色，低置信度红色
        cv2.polylines(vis, [pts], True, color, 2)
        text = d.get("text", "")
        if text:
            x = int(bbox[0][0])
            y = int(bbox[0][1]) - 6
            cv2.putText(vis, f"{text}({conf:.0%})", (x, max(y, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


class OCRService:
    """OCR 识别服务"""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from ocr import OCREngine
            self._engine = OCREngine(backend="paddleocr_seal")
            self._engine.initialize()
        return self._engine

    def recognize_fuzhou(self, image_path: str) -> dict:
        """福州模式识别"""
        engine = self._get_engine()
        from ocr import analyze_seal_file
        result = analyze_seal_file(image_path, ocr_engine=engine, enable_polar_unwrap=True)

        # 读取原图生成白底红字可视化
        orig = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        red_vis = _make_red_on_white(orig) if orig is not None else None

        preprocessed = result.get("preprocessed")
        unwrapped = result.get("unwrapped")
        details = result.get("ocr_details", [])

        return {
            "org_name": result.get("org_name", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("method", ""),
            "all_texts": result.get("all_texts", []),
            "candidates": result.get("candidates", []),
            "images": {
                "preprocessed": _img_to_base64(red_vis) if red_vis is not None else "",
                "unwrapped": _img_to_base64(unwrapped) if unwrapped is not None else "",
                "annotated": _img_to_base64(_draw_text_boxes(red_vis, details)) if red_vis is not None and details else "",
            },
        }

    def recognize_general(self, image_path: str) -> dict:
        """通用模式识别"""
        engine = self._get_engine()
        from ocr.tongyong import analyze_seal_file_general
        result = analyze_seal_file_general(image_path, ocr_engine=engine)

        warped = result.get("warped")
        details = result.get("ocr_details", [])

        # 对 warped 做颜色增强，得到白底黑字的预处理图
        enhanced = None
        preprocessed_b64 = ""
        if warped is not None:
            from ocr.tongyong.engine import adaptive_color_enhance
            try:
                enhanced = adaptive_color_enhance(warped)
                preprocessed_b64 = _img_to_base64(enhanced)
            except Exception:
                enhanced = warped
                preprocessed_b64 = _img_to_base64(warped)

        return {
            "org_name": result.get("org_name", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("method", ""),
            "all_texts": result.get("all_texts", []),
            "candidates": result.get("candidates", []),
            "shape_info": result.get("shape_info", {}),
            "images": {
                "preprocessed": preprocessed_b64,
                "annotated": _img_to_base64(_draw_text_boxes(enhanced, details)) if enhanced is not None and details else "",
            },
        }

    def recognize(self, image_path: str, mode: str = "general") -> dict:
        """统一接口"""
        if mode == "fuzhou":
            return self.recognize_fuzhou(image_path)
        return self.recognize_general(image_path)


_service = None


def get_ocr_service() -> OCRService:
    global _service
    if _service is None:
        _service = OCRService()
    return _service
