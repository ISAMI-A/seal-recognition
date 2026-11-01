import { useState, useRef, useEffect } from "react";
import * as api from "../api";
import type { BboxItem } from "../api";
import { collectDroppedFiles, filterByAccept, pickFiles } from "../upload";

interface TuneFile {
  id: string;
  name: string;
  path: string;
  url?: string;
}

const SESSION_KEY = "tune_session";
const FILES_KEY = "tune_files";
const ACCEPT = ".jpg,.jpeg,.png";

export default function ModelTune() {
  // 文件列表与 session 持久化到 localStorage（浏览器关闭后仍保留），
  // 并兼容从旧的 sessionStorage 一次性迁移
  const [session, setSession] = useState(() => {
    const ls = localStorage.getItem(SESSION_KEY);
    if (ls) return ls;
    const ss = sessionStorage.getItem(SESSION_KEY);
    if (ss) { localStorage.setItem(SESSION_KEY, ss); return ss; }
    return "";
  });
  const [files, setFiles] = useState<TuneFile[]>(() => {
    try {
      const ls = localStorage.getItem(FILES_KEY);
      if (ls) return JSON.parse(ls);
      const ss = sessionStorage.getItem(FILES_KEY);
      if (ss) { localStorage.setItem(FILES_KEY, ss); return JSON.parse(ss); }
    } catch { /* ignore */ }
    return [];
  });
  const [selectedFile, setSelectedFile] = useState<TuneFile | null>(null);
  const [boxes, setBoxes] = useState<BboxItem[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  // 正在框选的新矩形（图片像素坐标）
  const [drawing, setDrawing] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  // 拖动已有框的拖动状态
  const dragRef = useRef<{ index: number; startX: number; startY: number; origX: number; origY: number } | null>(null);
  // 拖拽角点调整框大小的状态
  const resizeRef = useRef<{
    index: number;
    corner: "tl" | "tr" | "bl" | "br";
    startX: number;
    startY: number;
    origX: number;
    origY: number;
    origW: number;
    origH: number;
  } | null>(null);
  // 始终保存最新标注，供鼠标/键盘事件后的自动保存读取
  const boxesRef = useRef<BboxItem[]>([]);
  // 撤销历史：记录每次修改前的标注快照
  const historyRef = useRef<BboxItem[][]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const MAX_HISTORY = 30;
  // 图片显示尺寸变化时强制重渲染，避免标注框错位
  const [, setImgVersion] = useState(0);
  // 拖拽上传高亮
  const [dragOver, setDragOver] = useState(false);
  // 文件列表勾选（用于批量删除 / 选择性导出）
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => { boxesRef.current = boxes; }, [boxes]);
  useEffect(() => { localStorage.setItem(SESSION_KEY, session); }, [session]);
  useEffect(() => { localStorage.setItem(FILES_KEY, JSON.stringify(files)); }, [files]);

  // 图片尺寸变化（窗口缩放、容器宽度变化）时强制重渲染，让框跟随图片
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() => setImgVersion((v) => v + 1));
    ro.observe(img);
    return () => ro.disconnect();
  }, [selectedFile]);

  // 静默自动保存标注（不弹提示）
  const autoSave = async (list?: BboxItem[]) => {
    if (!session || !selectedFile) return;
    try {
      await api.saveAnnotations(session, selectedFile.id, list ?? boxesRef.current);
    } catch { /* 静默失败，不打断操作 */ }
  };

  // 记录修改前的标注快照（供撤回）
  const pushHistory = (before: BboxItem[]) => {
    historyRef.current.push(before);
    if (historyRef.current.length > MAX_HISTORY) historyRef.current.shift();
    setCanUndo(true);
  };

  // 撤回上一步标注操作
  const handleUndo = () => {
    const prev = historyRef.current.pop();
    if (!prev) return;
    setBoxes(prev);
    setSelectedIdx(null);
    setDrawing(null);
    autoSave(prev);
    setCanUndo(historyRef.current.length > 0);
  };

  const handleUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;
    const res = await api.uploadTuneImages(files, session || undefined);
    setSession(res.session);
    setFiles((prev) => [...prev, ...res.files]);
  };

  const handleDeleteFile = async (f: TuneFile) => {
    if (session) {
      try { await api.deleteTuneFile(session, f.id); } catch { /* ignore */ }
    }
    setFiles((prev) => prev.filter((x) => x.id !== f.id));
    setSelectedIds((prev) => { const next = new Set(prev); next.delete(f.id); return next; });
    if (selectedFile?.id === f.id) {
      setSelectedFile(null);
      setBoxes([]);
    }
  };

  // 切换单个文件的勾选状态
  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // 批量删除勾选的文件
  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    if (session) {
      for (const id of selectedIds) {
        try { await api.deleteTuneFile(session, id); } catch { /* ignore */ }
      }
    }
    setFiles((prev) => prev.filter((f) => !selectedIds.has(f.id)));
    if (selectedFile && selectedIds.has(selectedFile.id)) {
      setSelectedFile(null);
      setBoxes([]);
    }
    setSelectedIds(new Set());
  };

  // 切换图片时避免异步乱序：仅采纳最后一次选择的请求结果
  const selectSeq = useRef(0);
  const selectFile = async (f: TuneFile) => {
    const seq = ++selectSeq.current;
    setSelectedFile(f);
    setSelectedIdx(null);
    setDrawing(null);
    dragRef.current = null;
    if (session) {
      const ann = await api.getAnnotations(session, f.id);
      if (seq !== selectSeq.current) return; // 已被更新的选择覆盖，丢弃过期结果
      setBoxes(ann.boxes || []);
    }
  };

  const handleSave = async () => {
    if (!session || !selectedFile) return;
    await api.saveAnnotations(session, selectedFile.id, boxes);
    alert("标注已保存");
  };

  // 清空当前图片的全部标注框（不弹确认，可撤回；同步保存）
  const handleClearAll = () => {
    if (boxes.length === 0) return;
    pushHistory(boxesRef.current); // 记录清空前状态
    setBoxes([]);
    setSelectedIdx(null);
    autoSave([]); // 清空后同步保存到后端
  };

  // 上一张 / 下一张
  const currentIdx = selectedFile ? files.findIndex((f) => f.id === selectedFile.id) : -1;
  const goPrev = () => {
    if (currentIdx <= 0) return;
    selectFile(files[currentIdx - 1]);
  };
  const goNext = () => {
    if (currentIdx < 0 || currentIdx >= files.length - 1) return;
    selectFile(files[currentIdx + 1]);
  };

  // 鼠标事件 → 图片像素坐标
  const toImageCoords = (e: React.MouseEvent) => {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * img.naturalWidth,
      y: ((e.clientY - rect.top) / rect.height) * img.naturalHeight,
    };
  };

  // 按下：命中已有框则开始拖动，否则开始框选
  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const pos = toImageCoords(e);
    if (!pos) return;
    // 从最上层（最后一个）开始找命中的框
    const hit = (() => {
      for (let i = boxes.length - 1; i >= 0; i--) {
        const b = boxes[i];
        if (pos.x >= b.x && pos.x <= b.x + b.width && pos.y >= b.y && pos.y <= b.y + b.height) return i;
      }
      return -1;
    })();
    if (hit >= 0) {
      pushHistory(boxesRef.current); // 记录拖动前状态
      setSelectedIdx(hit);
      dragRef.current = {
        index: hit,
        startX: pos.x,
        startY: pos.y,
        origX: boxes[hit].x,
        origY: boxes[hit].y,
      };
    } else {
      setSelectedIdx(null);
      setDrawing({ x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
    }
  };

  // 开始拖拽角点调整框大小
  const startResize = (index: number, corner: "tl" | "tr" | "bl" | "br", e: React.MouseEvent) => {
    e.stopPropagation();
    const pos = toImageCoords(e);
    const b = boxesRef.current[index];
    if (!pos || !b) return;
    pushHistory(boxesRef.current); // 记录调整大小前状态
    resizeRef.current = {
      index,
      corner,
      startX: pos.x,
      startY: pos.y,
      origX: b.x,
      origY: b.y,
      origW: b.width,
      origH: b.height,
    };
  };

  // 移动：调整框大小 / 拖动框更新位置 / 更新框选矩形
  const onMouseMove = (e: React.MouseEvent) => {
    const pos = toImageCoords(e);
    if (!pos) return;
    const rz = resizeRef.current;
    if (rz) {
      const dx = pos.x - rz.startX;
      const dy = pos.y - rz.startY;
      let nx = rz.origX, ny = rz.origY, nw = rz.origW, nh = rz.origH;
      if (rz.corner.includes("l")) { nx = rz.origX + dx; nw = rz.origW - dx; }
      if (rz.corner.includes("r")) { nw = rz.origW + dx; }
      if (rz.corner.includes("t")) { ny = rz.origY + dy; nh = rz.origH - dy; }
      if (rz.corner.includes("b")) { nh = rz.origH + dy; }
      // 最小尺寸 10px，防止翻转
      if (nw < 10) { if (rz.corner.includes("l")) nx = rz.origX + rz.origW - 10; nw = 10; }
      if (nh < 10) { if (rz.corner.includes("t")) ny = rz.origY + rz.origH - 10; nh = 10; }
      setBoxes((prev) =>
        prev.map((b, i) =>
          i === rz.index ? { ...b, x: Math.round(nx), y: Math.round(ny), width: Math.round(nw), height: Math.round(nh) } : b,
        ),
      );
      return;
    }
    const drag = dragRef.current;
    if (drag) {
      const dx = pos.x - drag.startX;
      const dy = pos.y - drag.startY;
      setBoxes((prev) =>
        prev.map((b, i) => (i === drag.index ? { ...b, x: Math.round(drag.origX + dx), y: Math.round(drag.origY + dy) } : b)),
      );
    } else if (drawing) {
      setDrawing({ ...drawing, x2: pos.x, y2: pos.y });
    }
  };

  // 松开：结束框选（面积足够则新增框），并自动保存
  const onMouseUp = () => {
    if (drawing) {
      const { x1, y1, x2, y2 } = drawing;
      const left = Math.min(x1, x2);
      const top = Math.min(y1, y2);
      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      if (w > 8 && h > 8) {
        pushHistory(boxesRef.current); // 记录新增框前状态
        const newBox: BboxItem = {
          x: Math.round(left),
          y: Math.round(top),
          width: Math.round(w),
          height: Math.round(h),
          class_id: 0,
          label: "seal",
        };
        const next = [...boxesRef.current, newBox];
        setBoxes(next);
        setSelectedIdx(next.length - 1);
        autoSave(next);
      } else {
        autoSave(); // 点击未形成框，仍保存当前状态
      }
      setDrawing(null);
    } else if (dragRef.current) {
      autoSave(); // 拖动框结束，自动保存新位置
    } else if (resizeRef.current) {
      autoSave(); // 调整大小结束，自动保存新尺寸
    }
    dragRef.current = null;
    resizeRef.current = null;
  };

  // 选中框后按 Delete 删除（自动保存）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Delete" && selectedIdx !== null) {
        pushHistory(boxesRef.current); // 记录删除前状态
        const next = boxesRef.current.filter((_, i) => i !== selectedIdx);
        setBoxes(next);
        setSelectedIdx(null);
        autoSave(next);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedIdx]);

  const handleExport = async (fmt: string) => {
    if (!session) return;
    // 导出前先把当前标注保存，避免导出到旧数据或报"无标注数据"
    if (selectedFile) {
      try {
        await api.saveAnnotations(session, selectedFile.id, boxes);
      } catch { /* 保存失败不阻塞导出 */ }
    }
    // 勾选了文件则只导出选中项，否则导出全部
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
    const form = new FormData();
    form.append("session", session);
    form.append("fmt", fmt);
    if (ids?.length) form.append("file_ids", JSON.stringify(ids));
    const res = await fetch(`/api/tune/export`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      let msg = `导出失败（HTTP ${res.status}）`;
      try {
        const data = await res.json();
        if (data?.detail) msg = data.detail;
      } catch { /* ignore */ }
      alert(msg);
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fmt === "yolo" ? "annotations_yolo.zip" : "annotations.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <h2>模型调优</h2>
      <p style={{ color: "#666", marginTop: 4 }}>
        审核 YOLO 检测结果，修正标注框，导出训练数据
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
        style={{ marginTop: 16 }}
      >
        <p style={{ fontSize: 28, marginBottom: 8 }}>📷</p>
        <p><strong>点击上传</strong> 或拖拽文件/文件夹到此处</p>
        <p style={{ fontSize: 12, color: "#999" }}>支持 JPG、PNG（可多选，也可拖入文件夹批量上传）</p>
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
        <div style={{ display: "flex", gap: 16, marginTop: 16 }}>
          {/* Sidebar */}
          <div style={{ width: 220, flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
              <h4 style={{ margin: 0 }}>图片列表</h4>
              <label style={{ fontSize: 12, color: "#666", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={files.length > 0 && selectedIds.size === files.length}
                  onChange={() => {
                    if (selectedIds.size === files.length) setSelectedIds(new Set());
                    else setSelectedIds(new Set(files.map((f) => f.id)));
                  }}
                  style={{ width: 16, height: 16, cursor: "pointer", accentColor: "#1a73e8" }}
                />
                全选
              </label>
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
                maxHeight: "50vh",
                overflowY: "auto",
                paddingRight: 4,
              }}
            >
              {files.map((f) => (
                <div key={f.id} style={{ display: "flex", gap: 2, alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.has(f.id)}
                    onChange={() => toggleSelect(f.id)}
                    title="勾选（用于批量删除/选择性导出）"
                    style={{ width: 16, height: 16, cursor: "pointer", accentColor: "#1a73e8" }}
                  />
                  <button
                    className={`btn ${selectedFile?.id === f.id ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => selectFile(f)}
                    onDoubleClick={() => toggleSelect(f.id)}
                    title="单击查看标注，双击勾选/取消"
                    style={{ textAlign: "left", fontSize: 12, padding: "6px 8px", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                  >
                    {f.name}
                  </button>
                  <button
                    className="btn btn-danger"
                    onClick={() => handleDeleteFile(f)}
                    style={{ padding: "6px 8px", fontSize: 11 }}
                    title="删除"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
              <button
                className="btn btn-danger"
                onClick={handleDeleteSelected}
                disabled={selectedIds.size === 0}
                style={{ width: "100%" }}
              >
                删除选中{selectedIds.size > 0 ? `（${selectedIds.size}）` : ""}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => handleExport("yolo")}
                style={{ width: "100%" }}
                title={selectedIds.size > 0 ? "仅导出勾选的文件" : "导出全部文件"}
              >
                导出 YOLO 格式{selectedIds.size > 0 ? "（选中）" : ""}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => handleExport("json")}
                style={{ width: "100%" }}
                title={selectedIds.size > 0 ? "仅导出勾选的文件" : "导出全部文件"}
              >
                导出 JSON 格式{selectedIds.size > 0 ? "（选中）" : ""}
              </button>
            </div>
          </div>

          {/* Image + boxes */}
          <div style={{ flex: 1 }}>
            {selectedFile ? (
              <>
                <p style={{ color: "#999", fontSize: 12, marginBottom: 4 }}>
                  拖拽框选印章区域 · 拖动已有框调整位置 · 拖动角点调整大小 · 选中后按 Delete 删除
                </p>
                <div
                  className="tune-canvas"
                  onMouseDown={onMouseDown}
                  onMouseMove={onMouseMove}
                  onMouseUp={onMouseUp}
                >
                  <img
                    ref={imgRef}
                    src={selectedFile.url || undefined}
                    alt={selectedFile.name}
                    draggable={false}
                    onLoad={() => setImgVersion((v) => v + 1)}
                  />
                  {drawing && (() => {
                    const img = imgRef.current;
                    if (!img) return null;
                    const scaleX = img.clientWidth / img.naturalWidth;
                    const scaleY = img.clientHeight / img.naturalHeight;
                    return (
                      <div
                        className="annotation-drawing"
                        style={{
                          left: Math.min(drawing.x1, drawing.x2) * scaleX,
                          top: Math.min(drawing.y1, drawing.y2) * scaleY,
                          width: Math.abs(drawing.x2 - drawing.x1) * scaleX,
                          height: Math.abs(drawing.y2 - drawing.y1) * scaleY,
                        }}
                      />
                    );
                  })()}
                  {boxes.map((box, i) => {
                    const img = imgRef.current;
                    if (!img) return null;
                    const scaleX = img.clientWidth / img.naturalWidth;
                    const scaleY = img.clientHeight / img.naturalHeight;
                    return (
                      <div
                        key={i}
                        className={`annotation-box${i === selectedIdx ? " selected" : ""}`}
                        style={{
                          left: box.x * scaleX,
                          top: box.y * scaleY,
                          width: box.width * scaleX,
                          height: box.height * scaleY,
                        }}
                      >
                        <span className="box-label">{box.label || "seal"}</span>
                        {i === selectedIdx &&
                          (["tl", "tr", "bl", "br"] as const).map((corner) => (
                            <span
                              key={corner}
                              className="box-handle"
                              style={{
                                // 手柄是 annotation-box 的子元素，定位相对框自身（非图片绝对坐标）
                                left: (corner.includes("r") ? box.width : 0) * scaleX - 5,
                                top: (corner.includes("b") ? box.height : 0) * scaleY - 5,
                                cursor: corner === "tl" || corner === "br" ? "nwse-resize" : "nesw-resize",
                              }}
                              onMouseDown={(e) => startResize(i, corner, e)}
                            />
                          ))}
                        <span
                          className="box-delete"
                          title="删除该标注框"
                          onMouseDown={(e) => e.stopPropagation()}
                          onClick={(e) => {
                            e.stopPropagation();
                            pushHistory(boxesRef.current); // 记录删除前状态
                            const next = boxesRef.current.filter((_, j) => j !== i);
                            setBoxes(next);
                            if (selectedIdx === i) setSelectedIdx(null);
                            autoSave(next);
                          }}
                        >
                          ✕
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div style={{ textAlign: "center", padding: 48, color: "#999" }}>请选择一张图片查看标注</div>
            )}
          </div>
        </div>
      )}

      {/* 圆形悬浮按钮：图片右侧中央，跟随页面滚动 */}
      {selectedFile && (
        <div
          style={{
            position: "fixed",
            right: 16,
            top: "50%",
            transform: "translateY(-50%)",
            display: "flex",
            flexDirection: "column",
            gap: 12,
            zIndex: 100,
          }}
        >
          <button className="float-btn" title="上一张" onClick={goPrev} disabled={currentIdx <= 0}>◀</button>
          <button
            className="float-btn"
            title="下一张"
            onClick={goNext}
            disabled={currentIdx < 0 || currentIdx >= files.length - 1}
          >▶</button>
          <button className="float-btn" title="撤回上一步" onClick={handleUndo} disabled={!canUndo}>↩</button>
          <button className="float-btn" title="保存标注" onClick={handleSave}>💾</button>
          <button className="float-btn" title="清空标注" onClick={handleClearAll}>🗑</button>
        </div>
      )}
    </div>
  );
}
