/**
 * 上传工具 — 拖拽（含文件夹递归）收集文件、扩展名过滤、文件夹选择
 */

// 递归遍历拖拽的文件夹/文件，收集为 File 列表
function walkEntry(entry: FileSystemEntry, out: File[]): Promise<void> {
  return new Promise((resolve) => {
    if (entry.isFile) {
      (entry as FileSystemFileEntry).file(
        (f) => { out.push(f); resolve(); },
        () => resolve(),
      );
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const readBatch = () => {
        reader.readEntries(async (batch) => {
          if (!batch.length) { resolve(); return; }
          for (const e of batch) await walkEntry(e, out);
          readBatch(); // 目录内容较多时需分批读取
        }, () => resolve());
      };
      readBatch();
    } else {
      resolve();
    }
  });
}

/**
 * 从拖拽的 DataTransfer 中收集文件（支持拖入文件夹，递归展开）
 * 兼容不支持 FileSystemEntry 的浏览器（退化为 files 列表）
 */
export async function collectDroppedFiles(dt: DataTransfer): Promise<File[]> {
  const files: File[] = [];
  if (dt.items?.length) {
    const entries: FileSystemEntry[] = [];
    for (const item of Array.from(dt.items)) {
      const entry = item.webkitGetAsEntry?.();
      if (entry) entries.push(entry);
    }
    if (entries.length) {
      for (const entry of entries) await walkEntry(entry, files);
      return files;
    }
  }
  return Array.from(dt.files || []);
}

/** 按 accept 扩展名过滤文件（文件夹上传时会混入无关文件） */
export function filterByAccept(files: File[], accept?: string): File[] {
  if (!accept) return files;
  const exts = accept
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.startsWith("."));
  if (!exts.length) return files;
  return files.filter((f) => exts.some((ext) => f.name.toLowerCase().endsWith(ext)));
}

/**
 * 打开文件/文件夹选择框
 * @param accept 扩展名过滤，如 ".jpg,.jpeg,.png"
 * @param onFiles 选择结果回调
 * @param directory 是否选择整个文件夹
 */
export function pickFiles(accept: string, onFiles: (files: File[]) => void, directory = false) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  // 隐藏 input：用户取消选择时 onchange 不触发，若不隐藏会残留可见的"选择文件"控件
  input.style.display = "none";
  if (accept) input.accept = accept;
  if (directory) {
    // webkitdirectory 是 WebKit 私有属性，需用 setAttribute 确保生效（部分浏览器对属性赋值不识别）
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
  }
  input.onchange = () => {
    if (input.files) onFiles(Array.from(input.files));
    input.remove();
  };
  // 挂载到 DOM 再触发，确保文件/文件夹选择器正确渲染
  document.body.appendChild(input);
  input.click();
  // 兜底清理：取消选择时 onchange 不触发，定时移除残留节点
  setTimeout(() => input.remove(), 60_000);
}
