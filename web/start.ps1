# 一键启动印章识别 Web 工具
# ======================
# 使用方法：
#   1. 确保已安装所需的 Python 依赖和 Node.js 依赖
#   2. 在 PowerShell 中运行: .\web\start.ps1
#   3. 自动启动后端（默认端口 8000，被占用自动顺延 8001/8002...）+ 前端（端口 5173）
#   4. 浏览器自动打开 http://localhost:5173
#   5. 按 Ctrl+C 停止所有服务（自动结束进程树）

Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     印章识别系统 - Web 启动器       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 项目根目录（$PSScriptRoot 即 web/，其父目录就是项目根目录）
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# ─── 探测后端端口：默认 8000，被占用则自动顺延（8001/8002...8009） ───
$BACKEND_PORT = 8000
for ($p = 8000; $p -le 8009; $p++) {
    if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) {
        $BACKEND_PORT = $p
        break
    }
}
if ($BACKEND_PORT -ne 8000) {
    Write-Host "⚠ 端口 8000 被占用，自动改用端口 $BACKEND_PORT" -ForegroundColor Yellow
}
# 把最终端口传给前端，vite 代理据此转发 /api、/static
$env:WEB_BACKEND_PORT = "$BACKEND_PORT"

$backendProc = $null
$frontendProc = $null

try {
    # ─── 1. 启动后端 ───
    Write-Host "[1/2] 启动后端 (ocr_venv + FastAPI，端口 $BACKEND_PORT)..." -ForegroundColor Yellow
    $backendProc = Start-Process -FilePath "$ROOT\ocr_venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "$BACKEND_PORT", "--reload" `
        -WorkingDirectory $ROOT -NoNewWindow -PassThru

    # 等待后端就绪（最多 15 秒）
    # 用 curl.exe + 127.0.0.1：PS 5.1 的 Invoke-WebRequest 会把 localhost 解析成 IPv6(::1)，
    # 而 uvicorn 监听 IPv4(0.0.0.0)，导致连不上而误判超时；curl 更快更可靠
    $ready = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 2 "http://127.0.0.1:$BACKEND_PORT/api/health" 2>$null
            if ($code -eq "200") { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        Write-Host "⚠ 后端启动超时，仍继续启动前端（请查看上方日志）" -ForegroundColor Yellow
    }
    Write-Host "  ✓ 后端已启动: http://localhost:$BACKEND_PORT" -ForegroundColor Green

    # ─── 2. 启动前端 ───
    Write-Host "[2/2] 启动前端 (React + Vite)..." -ForegroundColor Yellow
    $npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npmCmd) {
        Write-Host "✕ 找不到 npm.cmd，请确认 Node.js 已安装" -ForegroundColor Red
        exit 1
    }
    $frontendProc = Start-Process -FilePath $npmCmd `
        -ArgumentList "run", "dev" `
        -WorkingDirectory "$ROOT\web\frontend" -NoNewWindow -PassThru

    Start-Sleep -Seconds 2
    Write-Host "  ✓ 前端已启动: http://localhost:5173" -ForegroundColor Green
    Write-Host ""
    Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  在浏览器中打开: http://localhost:5173" -ForegroundColor White
    Write-Host "  按 Ctrl+C 停止所有服务" -ForegroundColor White
    Write-Host "══════════════════════════════════════" -ForegroundColor Cyan

    # 自动打开浏览器
    Start-Process "http://localhost:5173"

    # 等待用户按 Ctrl+C（或任一服务进程退出）
    Write-Host ""
    Write-Host "（后端/前端日志输出在上方，按 Ctrl+C 停止）" -ForegroundColor DarkGray
    while ($true) {
        Start-Sleep -Seconds 1
        if ($backendProc.HasExited) {
            Write-Host "⚠ 后端进程已退出" -ForegroundColor Red
            break
        }
        if ($frontendProc.HasExited) {
            Write-Host "⚠ 前端进程已退出" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "正在停止服务..." -ForegroundColor Yellow
    # 用 taskkill /T 结束整个进程树（可杀 uvicorn --reload 的 watcher/worker 和 vite 子进程）
    if ($backendProc -and -not $backendProc.HasExited) {
        & taskkill /PID $backendProc.Id /T /F 2>$null | Out-Null
    }
    if ($frontendProc -and -not $frontendProc.HasExited) {
        & taskkill /PID $frontendProc.Id /T /F 2>$null | Out-Null
    }
    Start-Sleep -Milliseconds 500
    Write-Host "已停止" -ForegroundColor Green
}
