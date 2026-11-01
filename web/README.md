# 印章识别系统 - Web 端

基于 FastAPI + React 的印章识别 Web 工具，将已有的 YOLO 印章检测和 PaddleX OCR 识别能力通过浏览器界面呈现，支持批量检测、过程可视化、结果导出与模型调优。

## 项目背景

项目前期已通过命令行方式完成印章检测（YOLO）与文字识别（PaddleX seal_recognition）的完整能力建设：

- **YOLO 检测**：从 PDF/图片中检测并裁剪出印章区域
- **OCR 识别**：对印章图片进行文字识别，支持福州专区（红色圆形印章 + 单位匹配库）和通用专区（多颜色多形状）

但命令行方式存在操作门槛高、结果不直观、无法可视化识别过程等问题。本项目将以上能力封装为 Web 应用，通过浏览器即可完成从上传 → 检测 → 识别 → 导出的一站式流程。

## 项目应用

| 场景 | 说明 |
|---|---|
| **印章检测** | 上传 PDF/图片，YOLO 自动检测并裁剪印章，缩略图网格展示 |
| **OCR 识别** | 两种模式（福州/通用）切换，可视化预处理过程图、文字检测框、置信度 |
| **结果导出** | 识别结果导出 CSV / Excel，可勾选导出部分结果 |
| **人工修正** | 识别结果支持点击编辑，导出前可修正错误 |
| **模型调优** | 交互式标注框（框选/拖动/删除/清空），导出 YOLO 官方训练格式 / JSON 预标注文件 |

## 技术栈

### 后端（FastAPI）

| 技术 | 版本 | 用途 |
|---|---|---|
| Python | 3.11 | 运行环境 |
| FastAPI | ≥0.104 | Web API 框架 |
| Uvicorn | ≥0.24 | ASGI 服务器 |
| OpenCV | - | 图像处理、文字框绘制 |
| PaddleX | 3.7.2 | seal_recognition 印章识别模型 |
| openpyxl / pandas | - | Excel 导出 |

### 前端（React）

| 技术 | 版本 | 用途 |
|---|---|---|
| React | 19.x | UI 框架 |
| React Router | 7.x | 页面路由 |
| Vite | 8.x | 构建工具 |
| TypeScript | 6.x | 类型安全 |

### 虚拟环境说明

> 项目使用两个独立的虚拟环境，避免依赖冲突：

| 虚拟环境 | 用途 | 位置 |
|---|---|---|
| `ocr_venv` | OCR 识别 + **Web 后端** | `testAI/ocr_venv/` |
| `venv` | YOLO 检测（PyTorch） | `testAI/venv/` |

Web 后端运行在 `ocr_venv` 中，需要调用 YOLO 时通过**子进程**调用 `venv` 中的 Python 脚本。

## 目录结构

```
testAI/
├── ocr/                      # OCR 识别模块（核心算法）
│   ├── fuzhou/               # 福州专区（红色圆形 + 匹配库）
│   └── tongyong/             # 通用专区（多色多形状）
├── yolo/                     # YOLO 检测模块
│   ├── models/印章识别模型.pt  # YOLO 模型
│   └── launcher.py           # YOLO 命令行启动器
├── data/                     # 数据目录
│   ├── _web_uploads/         # Web 上传的临时文件
│   ├── _web_crops/           # YOLO 裁剪结果
│   └── _web_tune/            # 模型调优标注数据
├── ocr_venv/                 # OCR 虚拟环境（Web 后端运行于此）
├── venv/                     # YOLO 虚拟环境
└── web/                      # ⬅ 本项目
    ├── start.ps1             # 一键启动脚本
    ├── backend/              # FastAPI 后端
    │   ├── main.py           # 应用入口
    │   ├── routers/          # API 路由（ocr/yolo/tune）
    │   └── services/         # 业务服务（OCR/YOLO 调用封装）
    └── frontend/             # React 前端
        └── src/
            ├── pages/        # 页面（Home/YoloDetect/OcrRecognize/ModelTune）
            ├── api.ts        # API 客户端
            └── App.tsx       # 路由配置
```

## 启动方法

### 方法一：一键启动（推荐）

在项目根目录运行：

```powershell
.\web\start.ps1
```

脚本会自动：
1. 启动后端（`ocr_venv` + Uvicorn，默认端口 8000，被占用自动顺延 8001/8002...）
2. 启动前端（Vite dev server，端口 5173）
3. 自动打开浏览器

> 注意：后端端口默认 8000，若被 LabelU 等其他工具占用会自动顺延；前端通过 Vite 代理转发 `/api`、`/static` 到后端（端口由脚本通过环境变量 `WEB_BACKEND_PORT` 传给 vite）。

> 注意：`start.ps1` 含中文内容，必须保持 **UTF-8 with BOM** 编码，否则 Windows PowerShell 5.1 会报"字符串缺少终止符"等解析错误。

### 方法二：手动分步启动

**终端 1 - 启动后端**：

```powershell
# 激活 OCR 虚拟环境
.\ocr_venv\Scripts\Activate.ps1

# 启动 FastAPI 后端
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 - 启动前端**：

```powershell
cd web\frontend
npm install          # 首次运行需要
npm run dev
```

**浏览器访问**：http://localhost:5173

## 前端页面

| 路由 | 页面 | 功能 |
|---|---|---|
| `/` | 首页 | 功能卡片导航 |
| `/yolo` | YOLO 检测 | 上传文件 → 检测裁剪 → 缩略图网格 → 发送到 OCR |
| `/ocr` | OCR 识别 | 上传印章图 → 模式切换 → 识别 → 过程可视化 → 导出 |
| `/tune` | 模型调优 | 交互式框选标注（拖拽/移动/删除/清空）→ 导出 YOLO 训练格式 / JSON 预标注 |

## API 概览

```
POST   /api/yolo/upload         # 上传 PDF/图片
POST   /api/yolo/detect         # 启动 YOLO 检测
DELETE /api/yolo/files          # 删除上传文件
DELETE /api/yolo/crops          # 删除裁剪结果
GET    /api/yolo/crops/:id/download  # 打包下载裁剪图

POST   /api/ocr/upload          # 上传印章图片
POST   /api/ocr/recognize       # 单张识别（mode: fuzhou/general）
DELETE /api/ocr/files           # 删除上传图片
POST   /api/ocr/export          # 导出 CSV/Excel

POST   /api/tune/upload           # 上传标注图片
DELETE /api/tune/files            # 删除上传文件
GET    /api/tune/annotations/:session/:file  # 获取单图标注
PUT    /api/tune/annotations/:session/:file  # 保存单图标注
POST   /api/tune/export           # 导出标注（fmt: json 预标注 / yolo 官方训练包）

POST   /api/data/clear           # 一键清空上传/裁剪/调优数据目录
```

## 模型调优与数据导出

### 标注交互

- **框选**：在图片上按住鼠标左键拖拽，生成新标注框（蓝色虚线预览，松开后生效）
- **移动**：按住框内任意位置拖动调整位置
- **删除**：点击框右上角 ✕ 按钮，或选中框后按 `Delete` 键
- **清空**：侧边栏"清空标注"按钮，一键清空当前图片的全部标注框
- 坐标以图片原始像素存储，与 YOLO 标注格式一致

### 导出 JSON（X-AnyLabeling 预标注格式）

导出单个 `annotations.json` 文件（数组形式），可直接导入 X-AnyLabeling 等标注工具：

```json
[
  {
    "sample_name": "公章样例.jpg",
    "config": { "polygonTool": [{ "color": "#0040ff", "key": "seal", "value": "seal" }] },
    "meta_data": { "width": 1600, "height": 1600, "rotate": 0 },
    "annotations": {
      "polygonTool": {
        "toolName": "polygonTool",
        "result": [
          { "order": 1, "id": "uuid", "label": "seal", "points": [[10,20],[110,20],[110,100],[10,100]] }
        ]
      }
    }
  }
]
```

- 矩形框自动转换为顺时针四点多边形
- `sample_name` 使用原始上传文件名（上传时记录于 session 目录的 `_files.json`）
- `meta_data.width/height` 读取图片真实尺寸

### 导出 YOLO（ultralytics 官方训练格式）

打包为 `annotations_yolo.zip`，解压后即可直接训练：

```
annotations_yolo.zip
├── images/  train/ + val/（原图）
├── labels/  train/ + val/（归一化 .txt：class cx cy w h）
└── data.yaml
```

- 数据按文件名哈希固定划分为 train 80% / val 20%（可复现）
- `data.yaml` 包含 `path`、`train`、`val`、`names` 类别映射
- 训练命令：`yolo train data=data.yaml`

### 导出注意

- 点击导出时会**自动保存当前标注**，无需先手动点"保存标注"
- 若 session 中无任何标注，后端返回 404，前端弹出错误提示（不会下载损坏文件）

## 依赖安装（新环境）

### 后端依赖

```powershell
.\ocr_venv\Scripts\pip install -r web\backend\requirements.txt
```

`requirements.txt` 内容：`fastapi`、`uvicorn`、`python-multipart`、`aiofiles`、`openpyxl`、`pandas`

### 前端依赖

```powershell
cd web\frontend
npm install
```

## 常见问题

**Q: 为什么删除文件后文件夹里还有残留？**
后端删除接口会同步删除服务器上的文件；若文件被占用或网络异常，可能残留，刷新页面即可重新同步。

**Q: 页面切换后上传的文件还在吗？**
文件列表保存在浏览器 `sessionStorage` 中，同一标签页内切换页面不丢失；关闭标签页后清空。

**Q: YOLO 检测提示找不到虚拟环境？**
确认 `venv` 目录存在且位于项目根目录 `testAI/venv/`，`venv/Scripts/python.exe` 是有效解释器。

**Q: 修改了 OCR 算法代码为什么不生效？**
后端以 `--reload` 启动时自动加载改动；若未生效，手动重启后端并清空 `web/backend/__pycache__/`。

**Q: 点击导出提示"无标注数据"？**
导出前需先保存标注（导出接口也会自动保存当前标注）。若 session 中没有任何已保存标注，后端返回 404，前端会弹出错误提示，而不是下载损坏文件。

**Q: 为什么之前 YOLO 导出的归一化坐标不对？**
历史版本用 `cv2.imread` 读取图片尺寸，无法识别中文路径（如 `沈佳勇`），会退回默认 640×640。已改用 `cv2.imdecode + np.fromfile` 修复。
