"""
PDF 渲染模块
功能：将 PDF 文件的每一页渲染为 PNG 图片
依赖：PyMuPDF (fitz)
"""

import os
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def render_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    page_prefix: Optional[str] = None
) -> list[str]:
    """
    将单个 PDF 文件渲染为 PNG 图片
    
    参数:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        dpi: 分辨率，默认 300
        page_prefix: 文件名前缀，默认使用 PDF 文件名
    
    返回:
        生成的图片路径列表
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if page_prefix is None:
        page_prefix = pdf_path.stem
    
    # 打开 PDF
    doc = fitz.open(pdf_path)
    image_paths = []
    
    # 计算缩放比例（1 point = 1/72 inch）
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 渲染为 PNG
        pix = page.get_pixmap(matrix=matrix)
        
        # 生成文件名：前缀_pageXXX.png
        page_name = f"{page_prefix}_page{page_num + 1:03d}.png"
        image_path = output_dir / page_name
        
        pix.save(str(image_path))
        image_paths.append(str(image_path))
        print(f"已渲染: {page_name}")
    
    doc.close()
    return image_paths


def batch_render_pdf(
    pdf_dir: str,
    output_dir: str,
    dpi: int = 300
) -> dict[str, list[str]]:
    """
    批量渲染目录下所有 PDF 文件
    
    参数:
        pdf_dir: PDF 文件目录
        output_dir: 输出目录
        dpi: 分辨率
    
    返回:
        字典，键为 PDF 文件名，值为生成的图片路径列表
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    
    pdf_extensions = {'.pdf'}
    
    results = {}
    
    pdf_files = [f for f in pdf_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in pdf_extensions]
    
    if not pdf_files:
        print(f"警告: 目录 {pdf_dir} 中没有找到 PDF 文件")
        return results
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    for pdf_file in sorted(pdf_files):
        print(f"\n处理: {pdf_file.name}")
        try:
            image_paths = render_pdf_to_images(
                str(pdf_file),
                str(output_dir),
                dpi=dpi
            )
            results[pdf_file.name] = image_paths
        except Exception as e:
            print(f"错误: 处理 {pdf_file.name} 失败 - {e}")
            results[pdf_file.name] = []
    
    return results


if __name__ == "__main__":
    DEFAULT_PDF_DIR = r"data/pdf"
    DEFAULT_OUTPUT_DIR = r"data/pdf_rendered"
    DEFAULT_DPI = 300
    
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_DIR
    output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DPI
    
    print(f"PDF 目录: {pdf_dir}")
    print(f"输出目录: {output_dir}")
    print(f"DPI: {dpi}")
    print("-" * 50)
    
    results = batch_render_pdf(pdf_dir, output_dir, dpi)
    
    total_images = sum(len(imgs) for imgs in results.values())
    print("\n" + "=" * 50)
    print(f"渲染完成！")
    print(f"PDF 文件数: {len(results)}")
    print(f"生成图片数: {total_images}")
