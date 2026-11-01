import { Link } from "react-router-dom";

const cards = [
  {
    to: "/yolo",
    icon: "🎯",
    title: "YOLO 印章检测",
    desc: "上传 PDF 或图片，自动检测并裁剪印章图片。支持批量处理、结果预览与导出。",
  },
  {
    to: "/ocr",
    icon: "🔍",
    title: "OCR 文字识别",
    desc: "对印章图片进行文字识别。支持福州专区（红色圆形）和通用专区（多色多形状）两种模式，可视化识别过程。",
  },
  {
    to: "/tune",
    icon: "🛠",
    title: "模型调优",
    desc: "审核并修正 YOLO 检测结果，导出 YOLO/JSON 格式标注文件，用于模型微调。",
  },
];

export default function Home() {
  return (
    <div>
      <h2 style={{ marginTop: 12 }}>欢迎使用印章识别系统</h2>
      <p style={{ color: "var(--text-dim)", marginTop: 8 }}>
        选择下方功能模块开始操作
      </p>
      <div className="card-grid">
        {cards.map((c) => (
          <Link key={c.to} to={c.to} className="card">
            <div className="card-icon">{c.icon}</div>
            <h3>{c.title}</h3>
            <p>{c.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
