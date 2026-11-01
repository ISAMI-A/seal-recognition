import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState } from "react";
import Home from "./pages/Home";
import YoloDetect from "./pages/YoloDetect";
import OcrRecognize from "./pages/OcrRecognize";
import ModelTune from "./pages/ModelTune";
import "./App.css";

// 各页面使用的缓存键（清除旧文件时一并清空，避免残留列表）
const CACHE_KEYS = ["yolo_files", "yolo_crops", "ocr_files", "ocr_results", "tune_session", "tune_files"];

export default function App() {
  const [clearing, setClearing] = useState(false);

  const handleClear = async () => {
    if (!window.confirm("确定清除所有旧文件？\n将清空上传、裁剪和模型调优的全部数据（不可恢复）。")) return;
    setClearing(true);
    try {
      const res = await fetch("/api/data/clear", { method: "POST" });
      if (!res.ok) {
        let msg = `清除失败（HTTP ${res.status}）`;
        try {
          const data = await res.json();
          if (data?.detail) msg = data.detail;
        } catch { /* ignore */ }
        alert(msg);
        return;
      }
      // 清空前端缓存后刷新页面
      CACHE_KEYS.forEach((k) => {
        localStorage.removeItem(k);
        sessionStorage.removeItem(k);
      });
      location.reload();
    } catch {
      alert("清除失败，请确认后端已启动");
    } finally {
      setClearing(false);
    }
  };

  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <a href="/" className="app-title">
            印章识别系统
          </a>
          <nav className="app-nav">
            <a href="/yolo">YOLO 检测</a>
            <a href="/ocr">OCR 识别</a>
            <a href="/tune">模型调优</a>
          </nav>
          <button className="btn btn-danger clear-btn" onClick={handleClear} disabled={clearing}>
            {clearing ? "清理中..." : "清除旧文件"}
          </button>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/yolo" element={<YoloDetect />} />
            <Route path="/ocr" element={<OcrRecognize />} />
            <Route path="/tune" element={<ModelTune />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
