import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import * as api from "../api";
import { collectDroppedFiles, filterByAccept, pickFiles } from "../upload";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  ext: string;
}

interface CropResult {
  name: string;
  url: string;
  size: number;
}

interface DetectResult {
  crops: CropResult[];
  task_ids: string[];
}

const FILE_KEY = "yolo_files";
const CROP_KEY = "yolo_crops";
const ACCEPT = ".pdf,.jpg,.jpeg,.png";

function loadFromStorage<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveToStorage(key: string, data: unknown) {
  sessionStorage.setItem(key, JSON.stringify(data));
}

export default function YoloDetect() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<UploadedFile[]>(() => loadFromStorage(FILE_KEY, []));
  const [detectResult, setDetectResult] = useState<DetectResult | null>(() => loadFromStorage(CROP_KEY, null));
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [selectedCrops, setSelectedCrops] = useState<Set<number>>(new Set());
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { saveToStorage(FILE_KEY, files); }, [files]);
  useEffect(() => { saveToStorage(CROP_KEY, detectResult); }, [detectResult]);

  // ── Upload ──
  const handleUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;
    setError(null);
    const res = await api.uploadFiles(files);
    setFiles((prev) => [...prev, ...res.files]);
  };

  // ── Delete individual file ──
  const handleDeleteFile = async (id: string) => {
    try {
      await api.deleteFile(id);
    } catch { /* ignore */ }
    setFiles((prev) => prev.filter((f) => f.id !== id));
    setSelectedFiles((prev) => { const next = new Set(prev); next.delete(id); return next; });
  };

  // ── Batch delete files ──
  const handleClearFiles = async () => {
    if (files.length === 0) return;
    const toDelete = selectedFiles.size > 0
      ? Array.from(selectedFiles)
      : files.map((f) => f.id);
    try {
      await api.deleteFiles(toDelete);
    } catch { /* ignore */ }
    setFiles((prev) => prev.filter((f) => !toDelete.includes(f.id)));
    setSelectedFiles(new Set());
  };

  // ── Toggle file select ──
  const toggleFile = (id: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  // ── Detect ──
  const handleDetect = async () => {
    if (files.length === 0) return;
    setDetecting(true);
    setDetectResult(null);
    setError(null);
    setSelectedCrops(new Set());
    try {
      const res = await api.detectYolo(files.map((f) => f.id));
      if (res.success && res.crops.length > 0) {
        setDetectResult({ crops: res.crops, task_ids: res.task_ids });
      } else if (res.errors && res.errors.length > 0) {
        setError(res.errors.join("\n"));
      } else {
        setError("未检测到印章");
      }
    } catch (e) {
      setError("检测失败: " + (e as Error).message);
    } finally {
      setDetecting(false);
    }
  };

  // ── Batch delete crops ──
  const handleClearCrops = async () => {
    if (!detectResult) return;
    const indices = selectedCrops.size > 0 ? Array.from(selectedCrops) : detectResult.crops.map((_, i) => i);
    // Only delete if task_ids exist (otherwise just clear frontend)
    if (detectResult.task_ids.length > 0) {
      try {
        await api.deleteCrops(detectResult.task_ids);
      } catch { /* ignore */ }
    }
    setDetectResult(null);
    setSelectedCrops(new Set());
  };

  // ── Send selected to OCR ──
  const handleSendSelectedToOcr = async () => {
    if (!detectResult || selectedCrops.size === 0) return;
    try {
      const uploads: { id: string; name: string }[] = [];
      for (const idx of selectedCrops) {
        const c = detectResult.crops[idx];
        if (!c) continue;
        const resp = await fetch(c.url);
        const blob = await resp.blob();
        const file = new File([blob], c.name, { type: "image/jpeg" });
        const res = await api.uploadOcrImages([file]);
        uploads.push(...res.files);
      }
      if (uploads.length > 0) {
        navigate(`/ocr?fileId=${uploads[0].id}&batch=${uploads.map(f => f.id).join(",")}`);
      }
    } catch (e) {
      alert("发送失败: " + (e as Error).message);
    }
  };

  // ── Toggle crop select ──
  const toggleCrop = (idx: number) => {
    setSelectedCrops((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      return next;
    });
  };

  // ── Send to OCR ──
  const handleSendToOcr = async (cropUrl: string, cropName: string) => {
    // Download the crop image and re-upload via OCR endpoint
    try {
      const resp = await fetch(cropUrl);
      const blob = await resp.blob();
      const file = new File([blob], cropName, { type: "image/jpeg" });
      const ocrRes = await api.uploadOcrImages([file]);
      navigate(`/ocr?fileId=${ocrRes.files[0].id}`);
    } catch (e) {
      alert("发送失败: " + (e as Error).message);
    }
  };

  return (
    <div>
      <h2>YOLO 印章检测</h2>
      <p style={{ color: "#666", marginTop: 4 }}>
        上传 PDF 或图片文件，自动检测并裁剪印章
      </p>

      {/* Upload zone */}
      <div
        className={`upload-zone ${dragOver ? "hover" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={async (e) => {
          e.preventDefault();
          setDragOver(false);
          const fs = await collectDroppedFiles(e.dataTransfer);
          handleUpload(filterByAccept(fs, ACCEPT));
        }}
        onClick={() => pickFiles(ACCEPT, (fs) => handleUpload(filterByAccept(fs, ACCEPT)))}
        style={{ marginTop: 16 }}
      >
        <p style={{ fontSize: 28, marginBottom: 8 }}>📁</p>
        <p><strong>点击上传</strong> 或拖拽文件/文件夹到此处</p>
        <p style={{ fontSize: 12, color: "#999" }}>支持 PDF、JPG、PNG（可多选，也可拖入文件夹批量上传）</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => handleUpload(filterByAccept(Array.from(e.target.files || []), ACCEPT))}
        />
      </div>
      <p style={{ marginTop: 8 }}>
        <button
          className="btn btn-secondary"
          onClick={() => pickFiles(ACCEPT, (fs) => handleUpload(filterByAccept(fs, ACCEPT)), true)}
          style={{ fontSize: 12, padding: "4px 10px" }}
        >
          📂 选择文件夹批量上传
        </button>
      </p>

      {/* File list */}
      {files.length > 0 && (
        <>
          <div className="table-wrap" style={{ marginTop: 16 }}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: 40 }}>
                    <input
                      type="checkbox"
                      checked={selectedFiles.size === files.length}
                      onChange={() => {
                        if (selectedFiles.size === files.length) setSelectedFiles(new Set());
                        else setSelectedFiles(new Set(files.map((f) => f.id)));
                      }}
                    />
                  </th>
                  <th>文件名</th>
                  <th>大小</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.id}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selectedFiles.has(f.id)}
                        onChange={() => toggleFile(f.id)}
                      />
                    </td>
                    <td>{f.name}</td>
                    <td>{(f.size / 1024).toFixed(1)} KB</td>
                    <td>
                      <button
                        className="btn btn-danger"
                        onClick={() => handleDeleteFile(f.id)}
                        style={{ padding: "4px 12px", fontSize: 12 }}
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleDetect} disabled={detecting}>
              {detecting ? "检测中..." : "开始检测"}
            </button>
            <button
              className="btn btn-danger"
              onClick={handleClearFiles}
              disabled={files.length === 0}
            >
              {selectedFiles.size > 0 ? `删除选中 (${selectedFiles.size})` : "清空列表"}
            </button>
          </div>
        </>
      )}

      {/* Progress bar */}
      {detecting && (
        <div style={{ marginTop: 16, padding: 16, background: "#e8f0fe", borderRadius: 8 }}>
          <p style={{ color: "#1a73e8", marginBottom: 8 }}>⏳ YOLO 检测中，请稍候...</p>
          <div className="progress-bar">
            <div className="progress-bar-inner" />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ marginTop: 16, padding: 12, background: "#fce8e6", borderRadius: 8, color: "#c5221f" }}>
          ❌ {error}
        </div>
      )}

      {/* Crop results */}
      {detectResult && detectResult.crops.length > 0 && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 24 }}>
            <h3>检测结果（{detectResult.crops.length} 个印章）</h3>
            <label style={{ fontSize: 13, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selectedCrops.size === detectResult.crops.length}
                onChange={() => {
                  if (selectedCrops.size === detectResult.crops.length) setSelectedCrops(new Set());
                  else setSelectedCrops(new Set(detectResult.crops.map((_, i) => i)));
                }}
                style={{ marginRight: 4 }}
              />
              全选
            </label>
          </div>
          <div className="crop-grid">
            {detectResult.crops.map((c, i) => (
              <div
                key={i}
                className="crop-item"
                style={{
                  borderColor: selectedCrops.has(i) ? "#1a73e8" : undefined,
                  background: selectedCrops.has(i) ? "#e8f0fe" : undefined,
                }}
              >
                <div style={{ position: "relative" }}>
                  <input
                    type="checkbox"
                    checked={selectedCrops.has(i)}
                    onChange={() => toggleCrop(i)}
                    style={{ position: "absolute", top: 4, left: 4, zIndex: 1 }}
                  />
                  <img
                    src={c.url}
                    alt={c.name}
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                </div>
                <div className="crop-name">{c.name}</div>
                <button
                  className="btn btn-primary"
                  style={{ marginTop: 6, padding: "2px 8px", fontSize: 11, width: "100%" }}
                  onClick={() => handleSendToOcr(c.url, c.name)}
                >
                  发送到 OCR
                </button>
              </div>
            ))}
          </div>
          <div className="btn-group">
            {selectedCrops.size > 0 && (
              <button className="btn btn-primary" onClick={handleSendSelectedToOcr}>
                发送选中到 OCR（{selectedCrops.size}）
              </button>
            )}
            <button className="btn btn-danger" onClick={handleClearCrops}>
              {selectedCrops.size > 0 ? `删除选中 (${selectedCrops.size})` : "删除全部结果"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
