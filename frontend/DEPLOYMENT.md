# Deploying the GIC Intelligence Frontend to Vercel

The frontend is a Create React App (CRA) single-page app. It deploys as a static
build on Vercel and runs **fully without a backend** — falling back to a
client-side realtime simulator and mock data — so it always demos cleanly.

## Deploy steps

1. Push this repo to GitHub and import it in Vercel (**New Project**).
2. **Root Directory:** set to `frontend` (this folder).
3. **Framework Preset:** `Create React App` (auto-detected).
4. Build settings (already in `vercel.json`):
   - Build Command: `npm run build`
   - Output Directory: `build`
5. **Environment Variable:**
   - `REACT_APP_API_URL` = your hosted backend URL (e.g. `https://gic-api.onrender.com`).
   - Omit it to run in simulated/mock mode with no backend.
6. Deploy. The `rewrites` rule in `vercel.json` sends all routes to
   `/index.html`, so client-side routing (`/login`, `/app/*`) works on refresh.

## Backend hosting

Do **not** host the FastAPI backend on Vercel serverless — it needs persistent
processes for ML models and WebSocket (`/ws/market`) connections. Host it on a
persistent platform such as **Render**, **Railway**, or **Fly.io**, then point
`REACT_APP_API_URL` at it.

The frontend derives the WebSocket URL from `REACT_APP_API_URL`
(`https://` → `wss://`), so a single env var configures both REST and realtime.

## Local development

```bash
cd frontend
cp .env.example .env   # adjust REACT_APP_API_URL if needed
npm install
npm start
```
