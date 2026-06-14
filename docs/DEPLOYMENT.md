# GIC Financial Intelligence — Deployment Guide

This document covers local development, Vercel deployments for both the React
frontend and FastAPI backend, and Supabase database setup.

---

## 1. Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Backend

```bash
# Clone the repository
git clone <repo-url>
cd GIC_Financial_Intelligence

# Install Python dependencies (editable mode)
pip install -r requirements.txt
# OR using pyproject.toml
pip install -e .

# Copy environment file and set variables
cp .env.example .env
# Edit .env — set GIC_SECRET_KEY and optionally SUPABASE_URL / SUPABASE_KEY

# Start the FastAPI server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
OpenAPI docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend

# Install Node dependencies
npm install

# Copy environment file
cp .env.example .env
# Ensure REACT_APP_API_URL=http://localhost:8000

# Start the React dev server
npm start
```

The dashboard will be available at `http://localhost:3000`.

### Default Demo Credentials

| Username | Password  | Role  |
|----------|-----------|-------|
| admin    | admin123  | admin |
| user     | user123   | user  |

---

## 2. Vercel Frontend Deployment

The React SPA lives in `frontend/` and is deployed as a static site with
Vercel's create-react-app preset.

### Steps

1. Connect the repository to Vercel.
2. Set **Root Directory** to `frontend`.
3. Vercel auto-detects `package.json` and uses `npm run build`.
4. `frontend/vercel.json` is already configured with SPA rewrite rules.

### Environment Variables (Vercel UI → Settings → Environment Variables)

| Variable             | Value                              | Required |
|----------------------|------------------------------------|----------|
| `REACT_APP_API_URL`  | `https://your-backend.vercel.app`  | Yes      |

### Verify

```bash
curl https://your-frontend.vercel.app/health  # should redirect to index.html
```

---

## 3. Vercel Backend Deployment

The FastAPI backend is deployed as a Vercel serverless function via the
root-level `vercel.json` and `api/index.py`.

### Steps

1. Connect the repository root to a **separate** Vercel project (not the
   same project as the frontend).
2. Vercel reads `vercel.json` at the repo root:
   - Build: `@vercel/python` compiles `src/api/app.py`.
   - Routes: all requests are forwarded to the FastAPI ASGI app via `api/index.py`.
3. Set environment variables (see table below).

### File Layout

```
GIC_Financial_Intelligence/
├── vercel.json          ← backend Vercel config (root)
├── api/
│   └── index.py         ← ASGI entry point re-exporting `app`
├── requirements.txt     ← pip dependencies for @vercel/python
└── src/api/app.py       ← FastAPI application
```

### Environment Variables (Vercel UI)

| Variable          | Description                                      | Required |
|-------------------|--------------------------------------------------|----------|
| `GIC_SECRET_KEY`  | JWT signing secret — use a strong random string  | Yes      |
| `SUPABASE_URL`    | `https://<project-id>.supabase.co`               | Optional |
| `SUPABASE_KEY`    | Supabase service-role or anon key                | Optional |

### Verify

```bash
curl https://your-backend.vercel.app/health
# Expected: {"status": "ok", ...}
```

---

## 4. Supabase Setup

The application works without Supabase (uses local JSON store by default).
Supabase is opt-in for persistent storage and multi-user scale.

### 4.1 Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project.
2. Note your **Project URL** and **service-role key** from
   Settings → API.

### 4.2 Run Migrations

**Option A — Supabase CLI (recommended)**

```bash
# Install CLI
npm install -g supabase

# Link to your project
supabase link --project-ref <project-id>

# Apply migrations and seed
supabase db reset
```

This automatically runs `supabase/migrations/001_init.sql` then
`supabase/seed.sql`.

**Option B — Supabase SQL Editor**

1. Open the Supabase dashboard → SQL Editor.
2. Paste and run `supabase/migrations/001_init.sql`.
3. Paste and run `supabase/seed.sql`.

**Option C — psql**

```bash
psql "$DATABASE_URL" -f supabase/migrations/001_init.sql
psql "$DATABASE_URL" -f supabase/seed.sql
```

### 4.3 Update Environment Variables

Add the following to your backend `.env` or Vercel environment settings:

```env
SUPABASE_URL=https://<project-id>.supabase.co
SUPABASE_KEY=<service-role-key>
```

### 4.4 Schema Reference

The combined schema reference lives at `supabase/schema.sql`.  It documents
execution order and provides commented-out `\i` commands for manual runs.

---

## 5. Environment Variables Reference

| Variable             | Used By           | Description                                                 |
|----------------------|-------------------|-------------------------------------------------------------|
| `REACT_APP_API_URL`  | Frontend (React)  | Full URL of the deployed backend, e.g. `https://api.example.com` |
| `GIC_SECRET_KEY`     | Backend (FastAPI) | JWT signing secret for the auth layer                       |
| `SUPABASE_URL`       | Backend (FastAPI) | Supabase project URL (optional — app works without it)     |
| `SUPABASE_KEY`       | Backend (FastAPI) | Supabase service-role key (optional)                        |

---

## 6. Health Check Verification

After deployment, verify the system is healthy:

```bash
# 1. Backend health
curl https://your-backend.vercel.app/health

# 2. API root / model list
curl https://your-backend.vercel.app/models

# 3. Auth — obtain a token
curl -X POST https://your-backend.vercel.app/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}'

# 4. Protected endpoint — forecasts (use token from step 3)
curl https://your-backend.vercel.app/insights/feed \
  -H 'Authorization: Bearer <token>'

# 5. Frontend loads and connects to backend
# Open https://your-frontend.vercel.app in a browser.
# Log in with admin / admin123 — dashboard should populate.
```

---

## 7. Architecture Notes

- The backend runs as a **stateless** Vercel serverless function.
  The local JSON stores (`auth/users.json`, `data/audit/`) are read-only
  after build — use Supabase for persistent writes in production.
- The frontend is a fully static React SPA with no server-side rendering.
- WebSocket endpoints (`/ws/market`) require a long-running server and are
  **not** compatible with Vercel serverless.  For WebSocket support, deploy
  the backend on Railway, Render, Fly.io, or a VPS instead.
