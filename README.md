# 印章识别系统

从 PDF / 图片中自动检测印章区域，并识别印章上的单位名称。项目提供两套能力形态：

- **命令行流程**：YOLO 检测裁剪 → OCR 文字识别 → CSV/JSON 结果输出
- **Web 界面**：FastAPI + React，浏览器完成上传 → 检测 → 识别 → 人工修正 → 导出 → 标注调优

## 项目用途

纸质档案数字化、合同/票据归档、印章信息核查等场景中，印章上的单位名称需要人工逐个录入，效率低且易出错。本项目把「找印章」和「读印章」两件事自动化：

1. YOLO 从整页 PDF/图片中定位印章并裁剪；
2. OCR 识别印章上的弧形文字，还原出完整单位名称；
3. 结果可导出表格，供后续人工复核或入库。

识别分两种模式：

| 模式 | 适用场景 | 处理策略 |
|---|---|---|
| `fuzhou` 福州专区 | 红色圆形印章（本地业务场景） | 霍夫圆检测 + 极坐标展开 + 福州单位匹配库 / 形近字纠错 / 前缀补全 |
| `general` 通用专区 | 任意颜色、圆形/椭圆/矩形印章 | 自适应颜色增强 + 形状检测 + 仿射校正 + 文本碎片合并与清洗 |

## 功能特性

### 命令行

- **YOLO 检测**：支持从 PDF 直接渲染检测（300 DPI 可配），也支持对已有图片目录直接检测
- **OCR 识别**：单次批量处理整个目录，输出印章单位名称、置信度、识别策略
- **结果导出**：同时生成 CSV（含全部文本碎片，便于排查）与 JSON（简洁结构，便于程序读取）
- **过程可调**：`ocr/fuzhou/extract.py`、`ocr/tongyong/extract.py` 中的关键词与匹配库可自行扩充

### Web 端（`web/`）

- **YOLO 检测页**：上传 PDF/图片 → 检测裁剪 → 缩略图网格展示 → 一键发送到 OCR
- **OCR 识别页**：模式切换（福州/通用）→ 识别 → 可视化预处理图、文字检测框与置信度 → 结果可点击编辑 → 导出 CSV/Excel
- **模型调优页**：图片上交互式框选（拖拽 / 移动 / 删除 / 清空），导出 YOLO 官方训练格式或 X-AnyLabeling 预标注 JSON
- **一键清空**：清空 Web 上传、裁剪、调优三类临时数据

> 本文档不含界面截图，功能以上表为准。

## 技术栈

| 组件 | 技术 | 作用 |
|---|---|---|
| 印章检测 | YOLOv8（ultralytics） | 定位印章位置并裁剪出印章区域 |
| 文字识别 | PaddleX `seal_recognition` | 识别印章弧形文字（印章专用模型） |
| 图像处理 | OpenCV | 颜色增强、形状检测、仿射变换、文字框绘制 |
| PDF 渲染 | PyMuPDF (fitz) | PDF 按 DPI 渲染为图片 |
| 数值计算 | NumPy | 像素矩阵运算 |
| Web 后端 | FastAPI + Uvicorn | 提供上传/检测/识别/导出/标注接口 |
| Web 前端 | React 19 + Vite + TypeScript | 单页应用界面与过程可视化 |

### 虚拟环境（重要）

YOLO 与 OCR 的依赖互相冲突（PyTorch 版本不同），因此项目使用**两个独立虚拟环境**：

| 环境 | 位置 | 用途 |
|---|---|---|
| `ocr_venv` | `ocr_venv/` | OCR 识别 + **Web 后端**（PaddleX / OpenCV） |
| `venv` | `venv/` | YOLO 检测（ultralytics / PyTorch） |

Web 后端运行在 `ocr_venv` 中，需要 YOLO 时通过**子进程**调用 `venv` 里的 Python。

## 目录结构

```
testAI/
├── launcher.py                # OCR 统一启动器（命令行入口）
├── ocr/                       # OCR 识别模块
│   ├── __init__.py            # 共享导出
│   ├── batch.py               # 批量处理
│   ├── fuzhou/                # 福州专区：engine / extract / pipeline
│   ├── tongyong/              # 通用专区：engine / extract / pipeline
│   └── requirements.txt
├── yolo/                      # YOLO 检测模块
│   ├── launcher.py            # YOLO 统一启动器（pdf / detect 两种模式）
│   ├── detect.py              # 检测与裁剪实现
│   ├── predict.py             # 批量预测
│   ├── export_labelu.py       # 导出 labelU 标注格式
│   ├── visualize.py           # 检测结果可视化
│   ├── models/                # 模型权重目录（需自行放入 印章识别模型.pt）
│   ├── tools/render_pdf.py    # PDF → PNG 渲染
│   └── requirements.txt
├── data/                      # 数据目录（Git 忽略，需自行准备）
│   ├── pdf/                   # PDF 源文件
│   ├── crops/                 # YOLO 裁剪结果
│   └── results/               # OCR 结果输出
├── web/                       # Web 应用
│   ├── start.ps1              # 一键启动脚本
│   ├── backend/               # FastAPI 后端（main.py / routers / services）
│   └── frontend/              # React 前端（Vite）
├── ocr_venv/                  # OCR 虚拟环境（Git 忽略）
├── venv/                      # YOLO 虚拟环境（Git 忽略）
├── PROJECT_README.md          # 命令行流程详细说明（算法细节）
└── web/README.md              # Web 端详细说明（接口清单、调优导出格式）
```

## 环境要求

- Windows 10/11（脚本为 PowerShell，其他系统需手动执行等价命令）
- Python 3.11
- Node.js 18+（仅 Web 前端需要）
- NVIDIA GPU（可选但推荐）：YOLO 默认 `device="cuda:0"`，无 GPU 需自行改为 `cpu`

## 快速开始

### 1. 安装依赖

```powershell
# OCR 环境
python -m venv ocr_venv
.\ocr_venv\Scripts\Activate.ps1
pip install -r ocr\requirements.txt
# 首次运行 PaddleX 会自动下载模型（约 2GB），需要联网

# YOLO 环境（新开一个终端）
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r yolo\requirements.txt
```

### 2. 准备模型与数据

- 将训练好的印章检测模型放到 `yolo/models/印章识别模型.pt`（权重不随仓库提供）
- 把待处理的 PDF 放到 `data/pdf/`，或把待识别图片放到 `data/crops/`

### 3. 命令行流程

```powershell
# 第一步：YOLO 检测裁剪（PDF → 印章小图）
.\venv\Scripts\Activate.ps1
python yolo\launcher.py --mode pdf --input data/pdf --crop data/crops

# 第二步：OCR 识别
.\ocr_venv\Scripts\Activate.ps1
python launcher.py --mode general --input data/crops
```

也可直接改脚本顶部配置区后无参运行：`launcher.py` 的 `MODE / DATA_PATH / OUTPUT_DIR / OUTPUT_NAME`，`yolo/launcher.py` 的 `MODE / INPUT_DIR / CROP_DIR / DPI`。

### 4. Web 端

```powershell
.\web\start.ps1
```

脚本会自动探测后端端口（默认 8000，占用则顺延 8001…）、启动前端（5173）并打开浏览器。手动启动方式：

```powershell
# 终端 1：后端
.\ocr_venv\Scripts\Activate.ps1
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload

# 终端 2：前端
cd web\frontend
npm install
npm run dev
```

> `web/start.ps1` 含中文，必须保存为 **UTF-8 with BOM**，否则 Windows PowerShell 5.1 会报字符串解析错误。

## 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEB_BACKEND_PORT` | `8000` | 后端端口；`start.ps1` 自动探测后写入，Vite 代理据此转发 `/api`、`/static` |
| `WEB_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | 后端允许的跨域来源，逗号分隔 |

其余配置（模式、输入/输出目录、DPI）都在启动脚本顶部配置区，不使用硬编码的绝对路径。

## 输出格式

CSV 列：`文件名, 单位名称, 置信度, 全部文本, 识别方法`

| 列 | 说明 |
|---|---|
| 文件名 | 输入图片文件名 |
| 单位名称 | 最终识别结果 |
| 置信度 | OCR 置信度（百分比） |
| 全部文本 | OCR 识别出的所有文本碎片，`\|` 分隔 |
| 识别方法 | `seal_recognition` / `upscaled` / `fallback_general` / `none` / `error` |

JSON 与 CSV 内容一致，但去掉「全部文本」列，便于程序读取。

## 项目完成度与已知限制

**已完成并可用**

- 命令行检测 + 识别全流程（两种模式）
- Web 端四个页面：首页、YOLO 检测、OCR 识别、模型调优
- 结果导出（CSV / Excel）、人工修正、标注框导出（YOLO / X-AnyLabeling 格式）
- 识别过程可视化（预处理图、文字检测框、置信度）

**已知限制**

- 模型权重与数据集不随仓库提供，需自行训练或向作者索取；缺少模型时检测/识别无法运行
- 福州专区的匹配库（`KNOWN_ORGANIZATIONS` 等，约 50 条）是本地业务单位，换地区需自行维护
- 依赖两个本地虚拟环境，未提供 Docker / 云端部署方案
- 印章遮挡严重、低分辨率或非红色印章在福州模式下识别率会明显下降（此时请改用 `general` 模式）
- Web 端为开发模式运行（Vite dev server + `--reload`），未做生产构建与鉴权

## 常见问题

**Q：YOLO 检测提示「虚拟环境未找到」？**
确认 `venv/Scripts/python.exe` 存在且位于项目根目录。

**Q：修改 OCR 算法代码后 Web 不生效？**
后端以 `--reload` 启动会自动重载；仍不生效则手动重启并清空 `web/backend/__pycache__/`。

**Q：PaddleOCR 初始化报 `strides is not right`？**
通用 PaddleOCR 引擎只初始化一次，失败后自动跳过，不影响 `seal_recognition` 主流程。

**Q：模型缓存下载到哪？**
默认缓存到系统 PaddleX 缓存目录，首次运行自动下载；删除对应子目录可重新下载。
