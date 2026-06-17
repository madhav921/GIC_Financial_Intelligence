#!/usr/bin/env bash
# GIC Financial Intelligence — local dev launcher (Unix/Mac)
# Prerequisites: Python 3.10+, Node.js 18+
set -e

# ── .env ──────────────────────────────────────────────────────────────
[ -f .env ] || cp .env.example .env

# ── Python venv ───────────────────────────────────────────────────────
if [ ! -d venv ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "Installing Python dependencies..."
pip install -q -r requirements-local.txt

# ── Node deps ─────────────────────────────────────────────────────────
if [ ! -d frontend/node_modules ]; then
  echo "Installing Node dependencies..."
  (cd frontend && npm install --silent)
fi

# ── Required directories ──────────────────────────────────────────────
mkdir -p data/raw data/processed data/synthetic data/external data/parquet \
         logs/audit models/saved

echo ""
echo "  Backend  → http://localhost:8000"
echo "  API docs → http://localhost:8000/docs"
echo "  Frontend → http://localhost:3000"
echo "  Login:     admin / admin123   (or user / user123)"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# ── Start backend (background) ────────────────────────────────────────
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# ── Stop both on exit ─────────────────────────────────────────────────
cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Start frontend (foreground) ───────────────────────────────────────
cd frontend && npm start
