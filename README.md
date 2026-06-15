# GIC Financial Intelligence Platform

**AI-powered Plan-to-Perform engine for automotive OEM commodity risk.**

Translates commodity market signals into quantified EBIT impact, VaR-bounded risk, and hedge recommendations — in real time, with full ML explainability and immutable governance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/frontend-React-61DAFB.svg)](https://react.dev)

---

## Run Locally (New Machine Setup)

Everything — backend, frontend, auth, synthetic data — runs self-contained. No external database or API keys required.

### Option A — Docker (recommended, 1 command)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/madhav921/GIC_Financial_Intelligence
cd GIC_Financial_Intelligence
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

First build takes ~3–5 min (downloads base images + installs deps). Subsequent runs start in seconds.

### Option B — Native (2 commands)

Requires Python 3.10+ and Node.js 18+.

**Unix / Mac:**
```bash
git clone https://github.com/madhav921/GIC_Financial_Intelligence
cd GIC_Financial_Intelligence
chmod +x start.sh && ./start.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/madhav921/GIC_Financial_Intelligence
cd GIC_Financial_Intelligence
.\start.ps1
```

The scripts auto-create the Python venv, install all dependencies, and start both servers.

### Demo login

| Role | Username | Password | Access |
|------|----------|----------|--------|
| Admin | `admin` | `admin123` | Full — simulations, audit trail, exports |
| User | `user` | `user123` | Read-only dashboards, sandbox simulation |

---

## What It Does

| Capability | Detail |
|-----------|--------|
| **Commodity Forecasting** | SARIMAX + XGBoost ensemble, 12 commodities, 5-fold CV, regime-adaptive blending |
| **Conformal Prediction** | Provable ≥90% coverage intervals via ACI — no Gaussian assumption |
| **Change-Point Detection** | CUSUM + BOCPD (G7) — fires same month a regime shifts, not 6 months later |
| **Monte Carlo Simulation** | 10K sims, fat-tail t(df=5), 7 preset scenarios, VaR/CVaR decomposition |
| **Quantile VaR** | XGBoost 2.x joint quantile objective — asymmetric 5th/95th risk bands (G11) |
| **SHAP Attribution** | TreeSHAP per-commodity feature drivers fed to LLM for plain-English narrative |
| **Hedge Optimiser** | Portfolio-theory optimal h* replacing naive % rules |
| **Warranty Analytics** | Weibull failure modes, EV learning curve, accrual adequacy |
| **Real-Time Feed** | WebSocket `/ws/market` — mean-reverting tick, client simulator fallback |
| **RBAC Auth** | 20-permission matrix, Admin/User, HMAC-SHA256 JWT, audit trail |
| **Plan-to-Perform Waterfall** | EBIT variance decomposed by driver: volume / price / commodity / FX / overhead |
| **Open-Source LLM** | Ollama → HuggingFace flan-t5 → template cascade, swap to Claude API in one line |

---

## Pipeline Results (Synthetic Data)

| Metric | Value |
|--------|-------|
| Commodities trained | 12 / 12 |
| Full pipeline runtime | 15.3 s |
| Revenue base | £176 bn |
| VaR(95%) | £19.3 bn |
| CVaR(95%) | £8.3 bn |
| Risk decomposition | Commodity 66% / FX 26% / Demand 8% |
| Audit events per run | 20 |
| API routes | 31 REST + 1 WebSocket |
| Frontend build | 226.6 kB gzip |

---

## Architecture

```
orchestrator.py — GICOrchestrator.run_full_pipeline()
│
├── Layer 1 · Data          layers/layer1_data/controller.py
│   DataLayerController → O-U synthetic / CSV / Parquet → commodity, macro, sales, BOM, warranty
│
├── Layer 2 · Intelligence  layers/layer2_intelligence/controller.py
│   IntelligenceLayerController → SARIMAX+XGBoost, Hurst regime, CUSUM+BOCPD (G7),
│                                  conformal intervals, SHAP, quantile forecaster (G11)
│
├── Layer 3 · Financial     layers/layer3_financial/controller.py
│   FinancialLayerController → BOM-weighted COGS, P&L waterfall, scenario shocks
│
├── Layer 4 · Simulation    layers/layer4_simulation/controller.py
│   SimulationLayerController → Monte Carlo 10K, 7 presets, hedge optimiser, fan chart
│
└── Layer 5 · Governance    layers/layer5_governance/controller.py
    GovernanceLayerController → GICLLMEngine, AuditTrail, BiasTracker, ExplainabilityEngine
```

**Frontend:** React SPA · 11 pages · Recharts · Tailwind · Vercel-deployable  
**Backend:** FastAPI · 31 routes · Pydantic v2 · CORS · OpenAPI docs at `/docs`  
**Database:** Supabase PostgreSQL (migration scripts in `supabase/`)

---

## Project Structure

```
GIC_Financial_Intelligence/
├── orchestrator.py              # Root entry point — wires all 5 layers
├── layers/                      # Layer controllers (one per architectural layer)
│   ├── layer1_data/
│   ├── layer2_intelligence/
│   ├── layer3_financial/
│   ├── layer4_simulation/
│   └── layer5_governance/
├── src/
│   ├── api/                     # FastAPI app + 8 route modules
│   │   ├── app.py               # App factory + CORS + router registration
│   │   └── routes/              # auth, forecast, pnl, simulation, insights, intelligence, realtime, health
│   ├── data/                    # Data loading, feature engineering, synthetic generator
│   ├── models/                  # All ML models + SOTA modules
│   │   ├── commodity_forecast.py
│   │   ├── conformal.py         # ACI split-conformal (G5)
│   │   ├── explainability_shap.py # TreeSHAP (G6)
│   │   ├── change_point.py      # CUSUM + BOCPD (G7)
│   │   ├── quantile_forecast.py # XGBoost quantile VaR (G11)
│   │   ├── hedge_optimizer.py
│   │   └── warranty_model.py
│   ├── simulation/              # Monte Carlo, scenario engine
│   ├── governance/              # Audit trail, bias tracking, explainability
│   └── insights/                # InsightEngine, variance bridge, EWS, recommendations
├── auth/                        # RBAC — models, permissions, security, store, FastAPI deps
├── api/
│   └── index.py                 # Vercel ASGI entry point
├── frontend/                    # React SPA
│   ├── src/
│   │   ├── pages/               # 11 pages (Landing, Login, Executive, Commodity, ...)
│   │   ├── components/          # Charts, Layout, Insights, Realtime, Common
│   │   ├── auth/                # AuthContext, ProtectedRoute, PermissionGate
│   │   ├── context/             # RealtimeContext (singleton WebSocket)
│   │   ├── hooks/               # useRealtime
│   │   └── api/                 # client.js — 19 typed API methods
│   └── vercel.json              # SPA rewrite rules
├── supabase/
│   ├── migrations/001_init.sql  # Full schema — 5 tables, RLS, 13 indexes
│   └── seed.sql                 # Demo users + sample data
├── data/
│   ├── synthetic/               # Generated CSV files (commodity, macro, sales, warranty)
│   └── audit/                   # JSONL audit trail (append-only)
├── docs/                        # 14 documentation files (see below)
├── vercel.json                  # Backend Vercel config
└── requirements.txt
```

---

## Run Full Pipeline (Python only)

```python
from orchestrator import GICOrchestrator
engine = GICOrchestrator()
results = engine.run_full_pipeline(n_simulations=10_000)
# returns: layer1_data, layer2_intelligence, layer3_financial,
#          layer4_simulation, layer5_governance, pipeline_elapsed_seconds
```

---

## Deployment

### Frontend → Vercel
```bash
cd frontend
vercel deploy --prod
# Set REACT_APP_API_URL=https://your-backend.vercel.app
```

### Backend → Vercel (Serverless)
```bash
# root vercel.json already configured
vercel deploy --prod
```

### Database → Supabase
```bash
# In Supabase SQL Editor:
-- Run supabase/migrations/001_init.sql
-- Run supabase/seed.sql
```

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for full step-by-step guide with env vars.

---

## API Reference (Key Endpoints)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/auth/login` | — | Login → JWT token |
| `GET` | `/auth/demo-profiles` | — | Quick-login profiles |
| `GET` | `/pnl/annual` | User | KPI strip data |
| `POST` | `/forecast/commodity` | User | Single commodity forecast |
| `GET` | `/forecast/commodity-index` | User | BOM-weighted index |
| `POST` | `/simulation/scenario` | Admin | Monte Carlo run |
| `GET` | `/simulation/compare-presets` | User | 7-scenario comparison |
| `GET` | `/insights/feed` | User | Ranked InsightCards |
| `GET` | `/insights/variance-bridge` | User | EBIT waterfall |
| `GET` | `/insights/early-warning` | User | Risk score 0–100 |
| `GET` | `/insights/warranty/summary` | User | Warranty forecast |
| `GET` | `/intelligence/change-points/{commodity}` | Admin | CUSUM+BOCPD alert |
| `GET` | `/intelligence/quantile-var` | Admin | Asymmetric VaR bands |
| `WS` | `/ws/market` | — | Real-time market feed |

Full OpenAPI spec: `http://localhost:8000/docs`

---

## Documentation

| File | Contents |
|------|---------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Installation, first run, common tasks |
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Layer-by-layer design, module reference |
| [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md) | Algorithm details, API table, RBAC matrix |
| [docs/WHY_HOW_IMPACT.md](docs/WHY_HOW_IMPACT.md) | Every feature: What / Why / How / Impact |
| [docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md) | ROI model, pricing, target customer profile |
| [docs/SELLING_DECK.md](docs/SELLING_DECK.md) | Evidence-based pitch, objection handling, demo script |
| [docs/COMPETITIVE_ANALYSIS.md](docs/COMPETITIVE_ANALYSIS.md) | vs Anaplan / Pigment / o9 / Kinaxis / SAP IBP |
| [docs/ROADMAP.md](docs/ROADMAP.md) | ✅/⏳/❌ checklist, P0/P1/P2 priorities, quick wins |
| [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) | 10-dimension scorecard vs SOTA and competitors |
| [docs/RESEARCH_WOWFACTORS.md](docs/RESEARCH_WOWFACTORS.md) | SOTA survey — N-BEATS, TFT, TimesFM, BOCPD, Time-LLM |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Vercel + Supabase deployment guide |
| [docs/OUTPUT_GUIDE.md](docs/OUTPUT_GUIDE.md) | How to read reports and dashboard pages |

---

## Frontend Pages

| Page | Route | Access | Key Features |
|------|-------|--------|-------------|
| Landing | `/` | Public | Value prop, stat band, feature grid |
| Login | `/login` | Public | Quick-demo buttons (Admin / User) |
| Executive Summary | `/app/executive` | User | Live KPI strip, risk gauge, top insights |
| Commodity Intelligence | `/app/commodity` | User | Forecast chart, MA/Bollinger overlays, SHAP drivers, heatmap |
| Financial P&L | `/app/pnl` | User | EBIT waterfall, segment bars, sensitivity slider |
| Scenario Simulation | `/app/simulation` | Admin (full) / User (sandbox) | Distribution histogram, tornado chart, VaR markers |
| Market Monitor | `/app/market` | User | Live tape, sparklines, FX panel |
| Insights Centre | `/app/insights` | User | Ranked InsightCards, £-quantified recommendations |
| Variance Bridge | `/app/variance` | User | Plan-to-Perform EBIT waterfall |
| Warranty Analytics | `/app/warranty` | User | Failure-mode breakdown, accrual adequacy |
| Governance | `/app/governance` | Admin (full) / User (summary) | Audit trail, bias table, LLM narratives |
| Data Explorer | `/app/data` | Admin only | Raw row preview, schema, quality metrics |

---

## RBAC Permissions

Admin has all 20 permissions. User has 9 read/sandbox permissions.

Key Admin-only: `RUN_SIMULATION`, `MANAGE_MODELS`, `TRIGGER_RETRAINING`, `VIEW_AUDIT_FULL`, `EXPORT_REPORTS`, `EDIT_SCENARIOS`, `MANAGE_THRESHOLDS`

Key User: `VIEW_DASHBOARD`, `VIEW_FORECASTS`, `VIEW_AGGREGATED_DATA`, `RUN_SANDBOX_SIMULATION`, `VIEW_AUDIT_SUMMARY`

---

## Branch

Active development branch: **`dev`**

```bash
git checkout dev
```
