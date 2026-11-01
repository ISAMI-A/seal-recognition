// 使用相对路径，由 Vite dev server 代理转发到后端（后端端口由 start.ps1 动态选择）
const API_BASE = "/api";

async function request(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...options?.headers },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API 错误 ${res.status}: ${text}`);
  }
  return res;
}

// ─── YOLO ───

export async function uploadFiles(files: File[]) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await request("/yolo/upload", { method: "POST", body: form });
  return res.json() as Promise<{ files: { id: string; name: string; size: number; ext: string }[] }>;
}

export async function deleteFile(fileId: string) {
  const form = new FormData();
  form.append("file_ids", JSON.stringify([fileId]));
  await request("/yolo/files", { method: "DELETE", body: form });
}

export async function detectYolo(fileIds: string[]) {
  const form = new FormData();
  form.append("file_ids", JSON.stringify(fileIds));
  const res = await request("/yolo/detect", { method: "POST", body: form });
  return res.json() as Promise<{
    success: boolean;
    crops: { name: string; url: string; size: number }[];
    count: number;
    task_ids: string[];
    errors?: string[];
  }>;
}

export async function deleteFiles(fileIds: string[]) {
  const form = new FormData();
  form.append("file_ids", JSON.stringify(fileIds));
  await request("/yolo/files", { method: "DELETE", body: form });
}

export async function deleteCrops(taskIds: string[]) {
  const form = new FormData();
  form.append("task_ids", JSON.stringify(taskIds));
  await request("/yolo/crops", { method: "DELETE", body: form });
}

// ─── OCR ───

export async function uploadOcrImages(files: File[]) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await request("/ocr/upload", { method: "POST", body: form });
  return res.json() as Promise<{ files: { id: string; name: string; size: number; url: string }[] }>;
}

export async function deleteOcrFiles(fileIds: string[]) {
  const form = new FormData();
  form.append("file_ids", JSON.stringify(fileIds));
  await request("/ocr/files", { method: "DELETE", body: form });
}

export async function recognizeOcr(fileId: string, mode: string = "general") {
  const form = new FormData();
  form.append("file_id", fileId);
  form.append("mode", mode);
  const res = await request("/ocr/recognize", { method: "POST", body: form });
  return res.json() as Promise<{
    file_id: string;
    mode: string;
    org_name: string;
    confidence: number;
    method: string;
    all_texts: string[];
    candidates: string[];
    images: { preprocessed: string; unwrapped: string };
    shape_info?: Record<string, unknown>;
  }>;
}

export async function exportOcr(data: Record<string, unknown>[], fmt: "csv" | "xlsx") {
  const form = new FormData();
  form.append("data", JSON.stringify(data));
  form.append("fmt", fmt);
  const res = await request("/ocr/export", { method: "POST", body: form });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ocr_results.${fmt}`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Tune ───

export async function uploadTuneImages(files: File[], session?: string) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  // 复用已有 session，保证多次上传的文件在同一目录
  if (session) form.append("session", session);
  const res = await request("/tune/upload", { method: "POST", body: form });
  return res.json() as Promise<{ session: string; files: { id: string; name: string; path: string; url: string }[] }>;
}

export async function deleteTuneFile(session: string, fileId: string) {
  const form = new FormData();
  form.append("data", JSON.stringify([{ session, file_id: fileId }]));
  await request("/tune/files", { method: "DELETE", body: form });
}

export async function getAnnotations(session: string, fileId: string) {
  const res = await request(`/tune/annotations/${session}/${fileId}`);
  return res.json() as Promise<{ boxes: BboxItem[] }>;
}

export async function saveAnnotations(session: string, fileId: string, boxes: BboxItem[]) {
  const form = new FormData();
  form.append("data", JSON.stringify({ boxes }));
  await request(`/tune/annotations/${session}/${fileId}`, { method: "PUT", body: form });
}

export interface BboxItem {
  x: number;
  y: number;
  width: number;
  height: number;
  class_id: number;
  label?: string;
}
