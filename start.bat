@echo off
setlocal enabledelayedexpansion
title GIC Financial Intelligence

echo ================================================
echo   GIC Financial Intelligence Platform
echo ================================================
echo.

:: ── Check Python ────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10+ is required but not found.
    echo Downloading installer...
    start "" "https://www.python.org/downloads/"
    echo After installing Python, re-run this file.
    pause & exit /b 1
)

:: ── Check Node.js ───────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js 18+ is required but not found.
    echo Downloading installer...
    start "" "https://nodejs.org/en/download/"
    echo After installing Node.js, re-run this file.
    pause & exit /b 1
)

:: ── .env ────────────────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" copy /y ".env.example" ".env" >nul
)

:: ── Python venv + deps ──────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo [1/4] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 ( echo Failed to create venv. & pause & exit /b 1 )
)
echo [2/4] Installing Python dependencies (first run: ~5 min)...
call venv\Scripts\pip.exe install -q -r requirements-local.txt
if errorlevel 1 ( echo pip install failed. & pause & exit /b 1 )

:: ── Node deps ───────────────────────────────────────────────────────
if not exist "frontend\node_modules" (
    echo [3/4] Installing Node.js dependencies...
    cd frontend && npm install --silent && cd ..
    if errorlevel 1 ( echo npm install failed. & pause & exit /b 1 )
)

:: ── Required directories ────────────────────────────────────────────
for %%d in (data\raw data\processed data\synthetic logs\audit models\saved) do (
    if not exist "%%d" mkdir "%%d" >nul 2>&1
)

:: ── Start backend ───────────────────────────────────────────────────
echo [4/4] Starting servers...
start "GIC-Backend" /min cmd /c "venv\Scripts\activate.bat && uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

:: Wait for backend to be ready
echo Waiting for backend to start...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto wait_backend

:: ── Start frontend ──────────────────────────────────────────────────
start "GIC-Frontend" /min cmd /c "cd frontend && npm start"

:: Wait for frontend dev server (React takes ~15-30s to compile)
echo Waiting for frontend to compile...
timeout /t 25 /nobreak >nul

:: ── Open browser ────────────────────────────────────────────────────
start "" "http://localhost:3000"

echo.
echo ================================================
echo   GIC is running!
echo.
echo   Dashboard  ^> http://localhost:3000
echo   API docs   ^> http://localhost:8000/docs
echo.
echo   Login: admin / admin123  (full access)
echo          user  / user123   (read-only)
echo ================================================
echo.
echo Press any key to STOP both servers and exit.
pause >nul

:: ── Cleanup ─────────────────────────────────────────────────────────
echo Stopping servers...
taskkill /f /fi "WINDOWTITLE eq GIC-Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq GIC-Frontend*" >nul 2>&1
