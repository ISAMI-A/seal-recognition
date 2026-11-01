"""
统一印章识别启动器
==================
运行前先激活虚拟环境：
  .\ocr_venv\Scripts\Activate.ps1

然后修改下面的 MODE 和 DATA_PATH 即可运行。

MODE 可选：
  "fuzhou"   - 福州专区（红色圆形印章，使用福州专属后处理）
  "general"  - 通用专区（各颜色/形状印章，无地区限定）
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════
# 用户配置区 - 修改下面几行即可
# ═══════════════════════════════════════════
MODE = "general"                 # "fuzhou" | "general"
DATA_PATH = r"data/test_general_102"       # 输入目录（相对路径直接写，Windows绝对路径要用 r"" 前缀）
OUTPUT_DIR = "data/results"     # 结果输出目录
OUTPUT_NAME = "results_seal_general"    # 输出文件名（不含扩展名）
# 示例：
#   DATA_PATH = "data/train_657"          # 相对路径
#   DATA_PATH = r"D:\xxx\testAI\data"    # 绝对路径（加 r 前缀防转义）
# ═══════════════════════════════════════════


def run_fuzhou(input_dir: str):
    """福州模式：默认 pipeline + 福州后处理"""
    print(f"[福州模式] 输入: {input_dir}")
    from ocr import OCREngine, analyze_seal_file

    engine = OCREngine(backend="paddleocr_seal")
    engine.initialize()

    files = sorted(Path(input_dir).glob("*.*"))
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"共 {len(files)} 张图片")

    results = []
    t_start = time.time()
    for i, f in enumerate(files, 1):
        try:
            result = analyze_seal_file(f, ocr_engine=engine, enable_polar_unwrap=True)
        except Exception as e:
            result = {"org_name": "", "confidence": 0.0, "all_texts": [], "method": "error"}
        results.append({
            "file_name": f.name,
            "org_name": result.get("org_name", ""),
            "confidence": result.get("confidence", 0.0),
            "all_texts": " | ".join(result.get("all_texts", [])),
            "method": result.get("method", ""),
        })
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {time.time()-t_start:.0f}s")

    csv_path = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.csv"
    json_path = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.json"
    _save_results(results, str(csv_path), str(json_path))
    recognized = sum(1 for r in results if r["org_name"])
    print(f"完成！{len(results)} 个，识别出 {recognized} 个（{recognized/len(results)*100:.1f}%）")


def run_general(input_dir: str):
    """通用模式：自适应颜色/形状 + 通用后处理"""
    print(f"[通用模式] 输入: {input_dir}")
    from ocr.tongyong import analyze_seal_file_general
    from ocr import OCREngine

    engine = OCREngine(backend="paddleocr_seal")
    engine.initialize()

    files = sorted(Path(input_dir).glob("*.*"))
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"共 {len(files)} 张图片")

    results = []
    t_start = time.time()
    for i, f in enumerate(files, 1):
        try:
            result = analyze_seal_file_general(f, ocr_engine=engine)
        except Exception as e:
            result = {"org_name": "", "confidence": 0.0, "all_texts": [], "method": "error"}
        results.append({
            "file_name": f.name,
            "org_name": result.get("org_name", ""),
            "confidence": result.get("confidence", 0.0),
            "all_texts": " | ".join(result.get("all_texts", [])),
            "method": result.get("method", ""),
        })
        if i % 10 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {time.time()-t_start:.0f}s")

    csv_gen = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.csv"
    json_gen = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}.json"
    _save_results(results, str(csv_gen), str(json_gen))
    recognized = sum(1 for r in results if r["org_name"])
    print(f"完成！{len(results)} 个，识别出 {recognized} 个（{recognized/len(results)*100:.1f}%）")


def _save_results(results, csv_path, json_path):
    """保存结果到CSV和JSON"""
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "单位名称", "置信度", "全部文本", "识别方法"])
        for r in results:
            w.writerow([
                r["file_name"], r["org_name"],
                f"{r['confidence']:.2%}",
                r.get("all_texts", ""),
                r["method"],
            ])
    with open(json_path, "w", encoding="utf-8") as f:
        clean = [{k: v for k, v in r.items() if k != "all_texts"} for r in results]
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {csv_path} / {json_path}")


if __name__ == "__main__":
    # 支持命令行参数覆盖
    import argparse
    parser = argparse.ArgumentParser(description="印章识别统一启动器")
    parser.add_argument("--mode", choices=["fuzhou", "general"],
                        default=MODE, help="运行模式")
    parser.add_argument("--input", "-i", default=DATA_PATH, help="输入目录或文件")
    args = parser.parse_args()

    if args.mode == "fuzhou":
        run_fuzhou(args.input)
    elif args.mode == "general":
        run_general(args.input)
