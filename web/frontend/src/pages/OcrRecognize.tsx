import { useState, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import * as api from "../api";
import { collectDroppedFiles, filterByAccept, pickFiles } from "../upload";

interface OcrFile {
  id: string;
  name: string;
  size: number;
  url: string;
}

interface OcrResult {
  file_id: string;
  file_name: string;
  mode: string;
  org_name: string;
  confidence: number;
  method: string;
  all_texts: string[];
  candidates: string[];
  images: { preprocessed: string; unwrapped: string; annotated: string };
}

const FILE_KEY = "ocr_files";
const RESULT_KEY = "ocr_results";

function load<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

export default function OcrRecognize() {
  const [searchParams] = useSearchParams();
  const [files, setFiles] = useState<OcrFile[]>(() => load(FILE_KEY, []));
  const [results, setResults] = useState<OcrResult[]>(() => load(RESULT_KEY, []));
  const [mode, setMode] = useState("general");
  const [recognizing, setRecognizing] = useState(false);
  const [selectedImg, setSelectedImg] = useState<string | null>(null);
  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [selectedResults, setSelectedResults] = useState<Set<number>>(new Set());
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const ACCEPT = ".jpg,.jpeg,.png";

  useEffect(() => { sessionStorage.setItem(FILE_KEY, JSON.stringify(files)); }, [files]);
  useEffect(() => { sessionStorage.setItem(RESULT_KEY, JSON.stringify(results)); }, [results]);

  // 从 YOLO 跳转：添加到文件列表，不自动识别
  useEffect(() => {
    const fileId = searchParams.get("fileId");
    if (fileId) {
      const batch = searchParams.get("batch");
      const ids = batch ? batch.split(",") : [fileId];
      const newFiles: OcrFile[] = ids.map((id) => ({
        id,
        name: searchParams.get("batch") ? "来自 YOLO 的裁剪图" : "来自 YOLO 的裁剪图",
        size: 0,
        url: `/static/uploads/${id}`,
      }));
      setFiles((prev) => {
        const exist = new Set(prev.map((f) => f.id));
        const add = newFiles.filter((f) => !exist.has(f.id));
        return [...prev, ...add];
      });
      // 清除 URL 参数，避免刷新后重复添加
      window.history.replaceState({}, "", "/ocr");
    }
  }, [searchParams]);

  const handleUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;
    const res = await api.uploadOcrImages(files);
    setFiles((prev) => [...prev, ...res.files]);
  };

  const handleDeleteFile = async (id: string) => {
    try { await api.deleteOcrFiles([id]); } catch { /* ignore */ }
    setFiles((prev) => prev.filter((f) => f.id !== id));
    setSelectedFileIds((prev) => { const n = new Set(prev); n.delete(id); return n; });
  };

  const handleClearFiles = async () => {
    const toDelete = selectedFileIds.size > 0
      ? Array.from(selectedFileIds) : files.map((f) => f.id);
    try { await api.deleteOcrFiles(toDelete); } catch { /* ignore */ }
    setFiles((prev) => prev.filter((f) => !toDelete.includes(f.id)));
    setSelectedFileIds(new Set());
  };

  const handleRecognize = async () => {
    if (files.length === 0) return;
    setRecognizing(true);
    setResults([]);  // 清空旧结果
    setSelectedResults(new Set());
    const newResults: OcrResult[] = [];
    try {
      for (const f of files) {
        const res = await api.recognizeOcr(f.id, mode);
        newResults.push({ ...res, file_name: f.name });
      }
      setResults(newResults);
    } catch (e) {
      alert("识别失败: " + (e as Error).message);
    } finally {
      setRecognizing(false);
    }
  };

  const handleExport = (fmt: "csv" | "xlsx") => {
    const toExport = selectedResults.size > 0
      ? results.filter((_, i) => selectedResults.has(i))
      : results;
    const data = toExport.map((r) => ({
      file_name: r.file_name,
      org_name: r.org_name,
      confidence: r.confidence,
      all_texts: r.all_texts,
      method: r.method,
    }));
    api.exportOcr(data, fmt);
  };

  const confPct = (v: number) => (v * 100).toFixed(1) + "%";

  return (
    <div>
      <h2>OCR 文字识别</h2>
      <div className="steps">
        <span className="step">🔵 当前使用 seal_recognition 印章专用模型</span>
        <span className="step">非印章图片可能无法正确识别</span>
      </div>

      {/* Mode switch */}
      <div className="mode-switch">
        <button className={`mode-btn ${mode === "general" ? "active" : ""}`} onClick={() => setMode("general")}>通用专区</button>
        <button className={`mode-btn ${mode === "fuzhou" ? "active" : ""}`} onClick={() => setMode("fuzhou")}>福州专区</button>
      </div>
      <p style={{ fontSize: 13, color: "#888", marginBottom: 12 }}>
        {mode === "general" ? "通用：支持多种颜色和形状，无地区匹配库" : "福州：仅红色圆形印章，有单位名称匹配库"}
      </p>

      {/* Upload */}
      <div
        className={`upload-zone ${dragOver ? "hover" : ""}`}
        onClick={() => pickFiles(ACCEPT, (fs) => handleUpload(filterByAccept(fs, ACCEPT)))}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={async (e) => {
          e.preventDefault();
          setDragOver(false);
          const fs = await collectDroppedFiles(e.dataTransfer);
          handleUpload(filterByAccept(fs, ACCEPT));
        }}
      >
        <p style={{ fontSize: 28, marginBottom: 8 }}>📄</p>
        <p><strong>点击上传</strong> 或拖拽文件/文件夹到此处</p>
        <p style={{ fontSize: 12, color: "#999" }}>支持 JPG、PNG（可多选，也可拖入文件夹批量上传）</p>
        <input ref={inputRef} type="file" multiple accept={ACCEPT} hidden onChange={(e) => handleUpload(filterByAccept(Array.from(e.target.files || []), ACCEPT))} />
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
                    <input type="checkbox" checked={selectedFileIds.size === files.length}
                      onChange={() => {
                        if (selectedFileIds.size === files.length) setSelectedFileIds(new Set());
                        else setSelectedFileIds(new Set(files.map((f) => f.id)));
                      }} />
                  </th>
                  <th>文件名</th>
                  <th>大小</th>
                  <th>预览</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.id}>
                    <td><input type="checkbox" checked={selectedFileIds.has(f.id)}
                      onChange={() => setSelectedFileIds((prev) => {
                        const n = new Set(prev);
                        if (n.has(f.id)) n.delete(f.id); else n.add(f.id);
                        return n;
                      })} /></td>
                    <td>{f.name}</td>
                    <td>{f.size > 0 ? (f.size / 1024).toFixed(1) + " KB" : "-"}</td>
                    <td>
                      {f.url && (
                        <img src={f.url} alt="" style={{ width: 48, height: 48, objectFit: "cover", borderRadius: 4, border: "1px solid #eee" }}
                          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                      )}
                    </td>
                    <td>
                      <button className="btn btn-danger" onClick={() => handleDeleteFile(f.id)} style={{ padding: "4px 12px", fontSize: 12 }}>删除</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleRecognize} disabled={recognizing}>
              {recognizing ? "识别中..." : `开始识别（${files.length} 张）`}
            </button>
            <button className="btn btn-danger" onClick={handleClearFiles}>
              {selectedFileIds.size > 0 ? `删除选中 (${selectedFileIds.size})` : "清空列表"}
            </button>
          </div>
        </>
      )}

      {recognizing && (
        <div style={{ marginTop: 16, padding: 16, background: "#e8f0fe", borderRadius: 8 }}>
          <p style={{ color: "#1a73e8", marginBottom: 8 }}>⏳ OCR 识别中...</p>
          <div className="progress-bar"><div className="progress-bar-inner" /></div>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 20 }}>
            <h3>识别结果（{results.length}）</h3>
            <label style={{ fontSize: 13, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={selectedResults.size === results.length}
                onChange={() => {
                  if (selectedResults.size === results.length) setSelectedResults(new Set());
                  else setSelectedResults(new Set(results.map((_, i) => i)));
                }}
                style={{ marginRight: 4 }}
              />
              全选
            </label>
          </div>
      {results.map((r, i) => (
        <div key={i} className="ocr-result" style={{ borderColor: selectedResults.has(i) ? "#1a73e8" : undefined }}>
          <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <input
              type="checkbox"
              checked={selectedResults.has(i)}
              onChange={() => setSelectedResults((prev) => {
                const n = new Set(prev);
                if (n.has(i)) n.delete(i); else n.add(i);
                return n;
              })}
              style={{ marginTop: 4 }}
            />
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div style={{ flex: 1 }}>
                  <div
                    className="org-name"
                    contentEditable
                    suppressContentEditableWarning
                    onBlur={(e) => {
                      const newName = e.currentTarget.textContent || "";
                      setResults((prev) => {
                        const next = [...prev];
                        next[i] = { ...next[i], org_name: newName };
                        return next;
                      });
                    }}
                    style={{ cursor: "text", borderBottom: "1px dashed #ccc" }}
                    title="点击编辑"
                  >
                    {r.org_name || "(未识别)"}
                  </div>
                  <div className="confidence">
                    置信度: {confPct(r.confidence)} | 方法: {r.method} | 模式: {r.mode}
                  </div>
                </div>
                <button
                  className="btn btn-danger"
                  onClick={() => {
                    setResults((prev) => prev.filter((_, j) => j !== i));
                    setSelectedResults((prev) => {
                      const n = new Set(prev);
                      n.delete(i);
                      return n;
                    });
                  }}
                  style={{ padding: "4px 10px", fontSize: 12, flexShrink: 0, marginLeft: 8 }}
                >
                  删除
                </button>
              </div>

              {r.all_texts.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <strong>全部文本:</strong>
                  <div style={{ marginTop: 4, color: "#555", fontSize: 14 }}>{r.all_texts.join(" | ")}</div>
                </div>
              )}

              {/* Process images */}
              {(r.images?.preprocessed || r.images?.annotated || r.images?.unwrapped) && (
                <div className="ocr-images">
                  {r.images.preprocessed && (
                    <div>
                      <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>预处理后（增强去噪）</div>
                      <img src={r.images.preprocessed} alt="preprocessed" onClick={() => setSelectedImg(r.images.preprocessed)} style={{ cursor: "pointer" }} />
                    </div>
                  )}
                  {r.images.annotated && (
                    <div>
                      <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>文字检测结果（绿色高置信度 → 红色低）</div>
                      <img src={r.images.annotated} alt="annotated" onClick={() => setSelectedImg(r.images.annotated)} style={{ cursor: "pointer", maxHeight: 200 }} />
                    </div>
                  )}
                  {r.images.unwrapped && (
                    <div>
                      <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>极坐标展开（圆形印章拉正）</div>
                      <img src={r.images.unwrapped} alt="unwrapped" onClick={() => setSelectedImg(r.images.unwrapped)} style={{ cursor: "pointer" }} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
      {/* Clear results */}
      <div className="btn-group">
        <button className="btn btn-danger" onClick={() => { setResults([]); setSelectedResults(new Set()); }}>
          清空结果
        </button>
        <button className="btn btn-secondary" onClick={() => handleExport("csv")}>
          {selectedResults.size > 0 ? `导出选中 CSV (${selectedResults.size})` : "导出 CSV"}
        </button>
        <button className="btn btn-secondary" onClick={() => handleExport("xlsx")}>
          {selectedResults.size > 0 ? `导出选中 Excel (${selectedResults.size})` : "导出 Excel"}
        </button>
      </div>
        </>
      )}

      {/* Image modal */}
      {selectedImg && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, cursor: "pointer" }}
          onClick={() => setSelectedImg(null)}>
          <img src={selectedImg} style={{ maxWidth: "90%", maxHeight: "90%", borderRadius: 8 }} alt="放大" />
        </div>
      )}
    </div>
  );
}
