# 印章识别系统

## 概述

端到端印章文字识别系统，支持两种模式：

1. **福州专区（fuzhou）** — 针对红色圆形印章优化，带福州专属后处理匹配库
2. **通用专区（general）** — 自适应颜色/形状，无地区限定

> OCR 识别使用 `ocr_venv` 虚拟环境。YOLO 检测有独立的 `venv` 环境和脚本，不走 launcher.py。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 印章检测 | YOLOv8 (ultralytics) |
| 文字识别 | PaddleX seal_recognition |
| 图像处理 | OpenCV |
| 数值计算 | NumPy |
| PDF 渲染 | PyMuPDF (fitz) |

---

## 文件目录结构

```
testAI/
├── launcher.py                   # 统一启动器（唯一入口）
├── ocr/                          # OCR 识别模块
│   ├── __init__.py               # 共享导出（福州引擎）
│   ├── batch.py                  # 批量处理
│   ├── test.py                   # 测试脚本
│   ├── fuzhou/                   # 福州专区
│   │   ├── engine.py             # 福州 OCR 引擎
│   │   ├── extract.py            # 福州后处理（含单位匹配库）
│   │   └── pipeline.py           # 福州识别流水线
│   └── tongyong/                 # 通用专区
│       ├── engine.py             # 通用 OCR 引擎（含形状/颜色检测）
│       ├── extract.py            # 通用后处理（无地区限定）
│       └── pipeline.py           # 通用识别流水线
├── yolo/                         # YOLO 检测模块
│   ├── __init__.py
│   ├── detect.py                 # YOLO 检测 + 裁剪
│   ├── predict.py                # 批量预测 + labelU 导出
│   ├── models/                   # YOLO 模型权重
│   │   └── 印章识别模型.pt
│   ├── datasets/                 # 训练数据集
│   └── tools/
│       └── render_pdf.py         # PDF → PNG 渲染工具
├── data/
│   ├── crops/                    # YOLO 裁剪后的印章图片
│   ├── results/                  # OCR 识别结果（CSV/JSON）
│   ├── train_657/                # 训练用印章图片（657张）
│   ├── test_general_102/         # 通用模式测试集（102张）
│   ├── pdf/                      # PDF 源文件
│   └── 标注数据.txt              # 手工标注标准答案
├── ocr_venv/                     # OCR 虚拟环境
├── venv/                         # YOLO 虚拟环境
└── PROJECT_README.md             # 本文件
```

---

## 安装部署（给新用户）

### 需要打包的文件

```
testAI/                          # 核心代码 ～1.2 MB
├── launcher.py                  # OCR 启动器
├── ocr/                         # OCR 识别模块
│   ├── fuzhou/                  #   福州专区
│   └── tongyong/                #   通用专区
├── yolo/                        # YOLO 检测模块
│   ├── launcher.py              #   YOLO 启动器
│   ├── detect.py                #   检测裁剪
│   ├── models/印章识别模型.pt    #   YOLO 模型（必备）
│   └── tools/render_pdf.py      #   PDF 渲染
├── PROJECT_README.md            # 本文件
├── ocr/requirements.txt         # OCR 环境依赖
└── yolo/requirements.txt        # YOLO 环境依赖

data/                            # 数据文件（如需重新跑）
├── pdf/                         #   PDF 源文件
├── train_657/                   #   训练用图片（可选）
└── 标注数据.txt                  #   标注答案（可选）
```

**不需要打包**（对方自己安装）：`ocr_venv/`（5.9 GB）、`venv/`（6.7 GB）、`yolo/datasets/`、`data/_pdf_rendered/`、`data/crops/`、`data/results/`、`__pycache__/`、`.idea/`

### 新用户安装步骤

```powershell
# 1. 创建 OCR 虚拟环境
python -m venv ocr_venv
.\ocr_venv\Scripts\Activate.ps1
pip install -r ocr\requirements.txt
# ⚠ paddlex 首次运行会自动下载模型（约 2GB），需要联网

# 2. 创建 YOLO 虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r yolo\requirements.txt
```

## 使用方法

### 1. 统一启动器（推荐）

修改 [launcher.py](file:///d:%5CUsers%5C沈佳勇%5CDesktop%5CtestAI%5Clauncher.py) 顶部配置区的三个参数即可运行：

```bash
.\ocr_venv\Scripts\Activate.ps1
python launcher.py
```

**配置区说明（launcher.py 第 25-28 行）：**

```python
MODE = "fuzhou"           # "fuzhou" | "general"
DATA_PATH = "data/crops"           # 输入目录
OUTPUT_DIR = "data/results"        # 结果输出目录
OUTPUT_NAME = "results_seal"       # 输出文件名（不含扩展名）
```

**MODE 可选值：**

| 模式 | 说明 |
|------|------|
| `fuzhou` | 福州专区（红色圆形印章，福州后处理） |
| `general` | 通用专区（各颜色/形状，无地区限定） |

也支持命令行参数覆盖：

```bash
.\ocr_venv\Scripts\Activate.ps1
python launcher.py --mode general --input data/test_general_102
```

### 2. 各模式运行

```bash
# 福州模式
python launcher.py --mode fuzhou --input data/crops

# 通用模式
python launcher.py --mode general --input data/test_general_102
```

---

## 通用模式（general）处理流程

```
输入图片
  → 1. 形状检测（圆形/椭圆/矩形）
  → 2. 仿射变换拉正（椭圆→圆形，矩形摆正）
  → 3. seal_recognition 识别
  → 4. 文本碎片合并（按圆形阅读顺序）
  → 5. 文本清理（去前导/尾部噪声、去符号、去重复）
  → 6. 单位名称有效性判断
  → 7. 输出结果
```

### 文本清理规则（clean_text）

| 规则 | 正则 | 示例 |
|------|------|------|
| 前导噪声 | `^[\dA-Za-z\-\(\)]+(?=[\u4e00-\u9fff])` | `21887V深圳` → `深圳`、`MC2新志市` → `新志市` |
| 尾部噪声 | `[\dA-Za-z\-\(\)]+$` | `委员会510` → `委员会` |
| 中间噪声 | `(?<=[\u4e00-\u9fff])[\dA-Za-z](?=[\u4e00-\u9fff])` | `有限4` → `有限` |
| 重复去重 | `(.)\1{2,}` → `\1\1` | `有限限公司` → `有限公司` |
| 符号过滤 | 预定义 PUNCTUATION_FILTER | 删除特殊符号 |

### 碎片合并规则（merge_seal_texts）

1. 按 `_seal_order` 排序（圆形阅读顺序）
2. 从第一个文本碎片开始，向后尝试合并
3. 合并后的文本包含 `ORG_KEYWORDS` 中的关键词则继续合并
4. 遇到 `STOP_WORDS`（发票章、合同章等）或超长数字串时停止合并

---

## 福州模式（fuzhou）处理流程

```
输入图片
  → 1. 霍夫圆检测 + 极坐标展开
  → 2. 多策略预处理 + seal_recognition 识别
  → 3. 福州专属后处理
     → 已知单位模糊匹配（KNOWN_ORGANIZATIONS）
     → 形近字纠错（SEAL_CORRECTION_MAP）
     → 前缀自动补全（PREFIX_MAP）
     → 编辑距离纠错
```

### 福州匹配库

| 数据结构 | 条目数 | 用途 |
|----------|--------|------|
| KNOWN_ORGANIZATIONS | ~50 | 完整单位名称列表 |
| KNOWN_ORG_ALIASES | ~25 | 短名→全称别名映射 |
| SEAL_CORRECTION_MAP | ~60 | 形近字纠错映射 |
| PREFIX_MAP | ~20 | 行政区划前缀补全 |

---

## YOLO 检测

YOLO 使用独立的 `venv` 虚拟环境，相关脚本在 `yolo/` 目录下：

| 脚本 | 用途 |
|------|------|
| `yolo/launcher.py` | **统一启动器（推荐）** |
| `yolo/detect.py` | 检测+裁剪（被启动器调用） |
| `yolo/predict.py` | 批量预测 + labelU 导出 |
| `yolo/tools/render_pdf.py` | PDF → PNG 渲染 |

### 统一启动器

修改 [yolo/launcher.py](file:///d:%5CUsers%5C沈佳勇%5CDesktop%5CtestAI%5Cyolo%5Clauncher.py) 顶部配置区即可：

```python
MODE = "pdf"                # "pdf" | "detect"
INPUT_DIR = "data/pdf"      # 输入：PDF目录（pdf模式）或图片目录（detect模式）
CROP_DIR = "data/crops"     # 裁剪印章输出目录
DPI = 300                   # PDF 渲染分辨率（仅 pdf 模式有效）
```

**MODE 可选值：**

| 模式 | 说明 |
|------|------|
| `pdf` | 从 PDF 开始：渲染 → 检测 → 裁剪，一键完成 |
| `detect` | 从已有图片开始：直接检测并裁剪 |

```bash
.\venv\Scripts\Activate.ps1

# 处理 data/pdf/ 里所有 PDF，印章裁剪到 data/crops/
python yolo\launcher.py

# 或处理已有图片
python yolo\launcher.py --mode detect --input data/some_images --crop data/crops
```

### 完整流程：YOLO 检测 → OCR 识别

```bash
# 第一步：YOLO 环境
.\venv\Scripts\Activate.ps1
python yolo\launcher.py

# 第二步：切换 OCR 环境
.\ocr_venv\Scripts\Activate.ps1
python launcher.py --mode general --input data/crops
```

---

## 输出格式

### CSV 格式

```
文件名,单位名称,置信度,全部文本,识别方法
0.jpg,北川羌族自治县永安镇长育村村民委员会,99.34%,北川羌族自治县永安镇长育村 | 村民委员会,seal_recognition
```

| 列 | 说明 |
|------|------|
| 文件名 | 输入图片文件名 |
| 单位名称 | 最终识别出的单位名称 |
| 置信度 | OCR 识别置信度 |
| 全部文本 | OCR 识别出的所有文本碎片（\| 分隔） |
| 识别方法 | 识别策略（seal_recognition / upscaled / fallback_general / none） |

### JSON 格式

同 CSV 数据，但去除"全部文本"列，便于程序读取。

---

## 模型缓存

模型文件缓存于 `D:\PADDLECTORCH_CACHE\`，首次运行自动下载。
如需重新下载，删除对应子目录即可。

---

## 常见问题

### 识别率提升

- 通用模式：在 `ocr/tongyong/extract.py` 的 `ORG_KEYWORDS` 中增加新关键词
- 福州模式：在 `ocr/fuzhou/extract.py` 的匹配库中增加新单位

### PaddleOCR 初始化报错

如果遇到 `Type of attribute: strides is not right` 错误：
- 通用 PaddleOCR 引擎只初始化一次，失败后自动跳过回退
- seal_recognition 不受影响

### 虚拟环境

| 环境 | 位置 | 用途 |
|------|------|------|
| ocr_venv | `ocr_venv/` | OCR 识别（PaddleX, PaddleOCR, OpenCV） |
| venv | `venv/` | YOLO 检测（ultralytics, PyTorch） |
