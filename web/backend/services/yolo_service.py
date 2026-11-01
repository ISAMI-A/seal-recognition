"""
YOLO 检测服务 — 通过子进程调用 venv 中的 YOLO
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # 项目根目录 testAI/
VENV_PYTHON = str(ROOT / "venv" / "Scripts" / "python.exe")
CROPS_DIR = ROOT / "data" / "_web_crops"


def _collect_crops(crop_dir: str) -> list[dict]:
    """收集裁剪目录中的图片，返回带 URL 路径的信息"""
    crop_path = Path(crop_dir)
    crops = []
    if not crop_path.exists():
        return crops

    for f in sorted(crop_path.iterdir()):
        if f.is_dir():
            continue
        ext = f.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png"):
            # 返回相对路径，前端通过 /static/crops/ 访问
            rel = f.relative_to(CROPS_DIR)
            crops.append({
                "name": f.name,
                "url": f"/static/crops/{rel.as_posix()}",
                "size": f.stat().st_size,
            })
    return crops


def _run_yolo_detect(input_dir: str, crop_dir: str) -> dict:
    """调用 YOLO launcher 检测图片中的印章"""
    if not os.path.exists(VENV_PYTHON):
        return {"success": False, "error": f"YOLO 虚拟环境未找到: {VENV_PYTHON}"}

    cmd = [
        VENV_PYTHON,
        str(ROOT / "yolo" / "launcher.py"),
        "--mode", "detect",
        "--input", input_dir,
        "--crop", crop_dir,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "YOLO 检测进程异常退出",
                "stderr": result.stderr,
                "stdout": result.stdout,
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "YOLO 检测超时（超过 300 秒）"}
    except Exception as e:
        return {"success": False, "error": f"YOLO 检测异常: {e}"}

    crops = _collect_crops(crop_dir)
    return {"success": True, "crops": crops, "count": len(crops)}


def _run_yolo_detect_pdf(pdf_dir: str, crop_dir: str, dpi: int = 300) -> dict:
    """调用 YOLO launcher 的 pdf 模式"""
    if not os.path.exists(VENV_PYTHON):
        return {"success": False, "error": f"YOLO 虚拟环境未找到: {VENV_PYTHON}"}

    cmd = [
        VENV_PYTHON,
        str(ROOT / "yolo" / "launcher.py"),
        "--mode", "pdf",
        "--input", pdf_dir,
        "--crop", crop_dir,
        "--dpi", str(dpi),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "YOLO PDF 检测进程异常退出",
                "stderr": result.stderr,
                "stdout": result.stdout,
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "YOLO PDF 检测超时（超过 600 秒）"}
    except Exception as e:
        return {"success": False, "error": f"YOLO PDF 检测异常: {e}"}

    crops = _collect_crops(crop_dir)
    return {"success": True, "crops": crops, "count": len(crops)}
