"""
验证脚本：在图片上画出检测框，可视化查看识别效果
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path

# 配置
IMAGES_DIR = r"D:\Users\沈佳勇\Desktop\train_images"
JSON_DIR = r"D:\Users\沈佳勇\Desktop\testAI\data\annotations"
OUTPUT_DIR = r"D:\Users\沈佳勇\Desktop\testAI\output\visualized"

def visualize_detection(image_name):
    """可视化单张图片的检测结果并保存"""
    # 构建路径
    image_path = os.path.join(IMAGES_DIR, image_name)
    json_path = os.path.join(JSON_DIR, f"{Path(image_name).stem}.json")
    
    # 读取 JSON
    if not os.path.exists(json_path):
        print(f"错误: JSON 文件不存在 {json_path}")
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 读取图片（支持中文路径）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None
    
    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")
    print(f"检测到 {len(data.get('seals', []))} 个印章")
    
    # 画框
    for i, seal in enumerate(data.get("seals", []), 1):
        x1, y1 = seal["x1"], seal["y1"]
        x2, y2 = seal["x2"], seal["y2"]
        conf = seal.get("confidence", 0)
        
        # 画矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 标签
        label = f"Seal {i}: {conf:.0%}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        print(f"  印章 {i}: ({x1},{y1}) -> ({x2},{y2}), 置信度: {conf:.0%}")
    
    # 保存到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"viz_{Path(image_name).stem}.jpg")
    cv2.imencode('.jpg', img)[1].tofile(output_path)
    print(f"已保存: {output_path}")
    return output_path

def batch_visualize(max_images=10):
    """批量可视化"""
    # 获取所有 JSON 文件
    json_files = sorted(Path(JSON_DIR).glob("*.json"))[:max_images]
    
    print(f"准备可视化 {len(json_files)} 张图片\n")
    print(f"标注结果将保存到: {OUTPUT_DIR}\n")
    
    for json_file in json_files:
        image_name = json_file.stem + ".jpg"
        print(f"\n{'='*50}")
        print(f"处理: {image_name}")
        print('='*50)
        
        visualize_detection(image_name)

if __name__ == "__main__":
    print("印章检测可视化工具")
    print("="*50)
    print("操作说明：")
    print("  - 按任意键：查看下一张")
    print("  - 按 ESC：退出")
    print("="*50)
    
    batch_visualize(max_images=10)
