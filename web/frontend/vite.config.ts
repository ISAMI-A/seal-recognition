import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 后端端口由 start.ps1 动态选择（默认 8000，被占用自动顺延），通过环境变量传给 vite
const backendTarget = `http://localhost:${process.env.WEB_BACKEND_PORT || "8000"}`;

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/static": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
