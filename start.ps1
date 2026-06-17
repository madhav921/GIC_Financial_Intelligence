# GIC Financial Intelligence — local dev launcher (Windows PowerShell)
# Prerequisites: Python 3.10+, Node.js 18+
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# ── .env ──────────────────────────────────────────────────────────────
if (-not (Test-Path "$Root\.env")) {
    Copy-Item "$Root\.env.example" "$Root\.env"
    Write-Host "Created .env from .env.example"
}

# ── Python venv ───────────────────────────────────────────────────────
if (-not (Test-Path "$Root\venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv "$Root\venv"
}
Write-Host "Installing Python dependencies..."
& "$Root\venv\Scripts\pip.exe" install -q -r "$Root\requirements-local.txt"

# ── Node deps ─────────────────────────────────────────────────────────
if (-not (Test-Path "$Root\frontend\node_modules")) {
    Write-Host "Installing Node dependencies..."
    Push-Location "$Root\frontend"
    npm install --silent
    Pop-Location
}

# ── Required directories ──────────────────────────────────────────────
$dirs = @(
    "$Root\data\raw", "$Root\data\processed", "$Root\data\synthetic",
    "$Root\data\external", "$Root\data\parquet",
    "$Root\logs\audit", "$Root\models\saved"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force $d | Out-Null
}

Write-Host ""
Write-Host "  Backend  -> http://localhost:8000"
Write-Host "  API docs -> http://localhost:8000/docs"
Write-Host "  Frontend -> http://localhost:3000"
Write-Host "  Login:     admin / admin123   (or user / user123)"
Write-Host ""
Write-Host "Starting backend in a new window. Close that window or press Ctrl+C here to stop."
Write-Host ""

# ── Start backend in a new window ────────────────────────────────────
$backendCmd = "& '$Root\venv\Scripts\uvicorn.exe' src.api.app:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; $backendCmd"

# ── Start frontend (foreground) ───────────────────────────────────────
Push-Location "$Root\frontend"
npm start
Pop-Location
