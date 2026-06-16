# GIC Financial Intelligence Platform

**AI-powered Plan-to-Perform engine for automotive OEM commodity risk.**

Translates commodity market signals into quantified EBIT impact, VaR-bounded risk, and hedge recommendations — in real time, with full ML explainability and immutable governance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/frontend-React-61DAFB.svg)](https://react.dev)

---

## Run Locally (New Machine Setup)

Everything — backend, frontend, auth, data — runs self-contained. No external database or API keys required.

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

First build takes ~3–5 min. Subsequent runs start in seconds.

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

### To use real-world live data

```bash
pip install -e ".[full]"         # yfinance, fredapi, ccxt, polars, scipy
python scripts/fetch_data.py     # ~2–3 min — fetches Yahoo Finance + FRED + CCXT
python scripts/train_models.py   # optional — re-trains on real data
uvicorn src.api.app:app --reload --port 8000
```

After `fetch_data.py` runs, all backend endpoints serve **real commodity prices, real FX rates, and real macro indicators**. Run it weekly to keep data fresh.

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
| **Change-Point Detection** | CUSUM + BOCPD — fires same month a regime shifts, not 6 months later |
| **Monte Carlo Simulation** | 10K sims, fat-tail t(df=5), 7 preset scenarios, VaR/CVaR decomposition, monthly fan chart |
| **Variance Decomposition** | MC-based attribution: commodity % / demand % / FX % of total EBIT variance |
| **Quantile VaR** | XGBoost 2.x joint quantile objective — asymmetric 5th/95th risk bands |
| **SHAP Attribution** | TreeSHAP per-commodity feature drivers fed to LLM for plain-English narrative |
| **Hedge Optimiser** | Portfolio-theory optimal h* replacing naive % rules |
| **Warranty Analytics** | Weibull failure modes, EV learning curve, accrual adequacy |
| **Real-Time Feed** | WebSocket `/ws/market` — mean-reverting tick seeded from Yahoo Finance data |
| **RBAC Auth** | 20-permission matrix, Admin/User, HMAC-SHA256 JWT, audit trail |
| **Plan-to-Perform Waterfall** | EBIT variance decomposed by driver: volume / price / commodity / FX / overhead |
| **Open-Source LLM** | Ollama → HuggingFace flan-t5 → template cascade |

---

## Data Sources

Real data by default — synthetic only as a last resort.

| Dataset | Source | Fallback |
|---------|--------|---------|
| 9/12 commodity prices | Yahoo Finance (7y monthly) | Ornstein-Uhlenbeck synthetic |
| 3/12 commodities (Rhodium, Polypropylene, ABS Resin) | Synthetic (no free source) | — |
| Macro indicators | FRED + Yahoo Finance proxies | Synthetic |
| FX rates (USD/GBP, EUR/USD, etc.) | Yahoo Finance | Synthetic |
| Market indices (S&P 500, VIX, Oil) | Yahoo Finance | Static reference |
| Sales & BOM data | Config-calibrated synthetic | Parquet/SAP/Salesforce ready |

See **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** for the full data flow, per-endpoint source table, and how to activate each source.

---

## Pipeline Results

| Metric | Value |
|--------|-------|
| Commodities modelled | 12 / 12 |
| Full pipeline runtime | ~15 s |
| Revenue base (JLR-calibrated) | £19.8B |
| EBIT | £1,401M (7.1% margin) |
| Monte Carlo VaR(95%) | ~£1.3B downside |
| Risk decomposition | Commodity ~62% / Demand ~23% / FX ~15% |
| Audit events per pipeline run | 20 |
| API routes | 33 REST + 1 WebSocket |
| Frontend build | <250 kB gzip |

---

## Architecture

```
orchestrator.py — GICOrchestrator.run_full_pipeline()
│
├── Layer 1 · Data          layers/layer1_data/controller.py
│   DataLayerController → data/raw/ (real) → data/synthetic/ (fallback)
│   commodity_prices, macro_indicators, sales_data, bom_data, warranty_data
│
├── Layer 2 · Intelligence  layers/layer2_intelligence/controller.py
│   IntelligenceLayerController → SARIMAX+XGBoost ensemble, Hurst regime,
│                                  CUSUM+BOCPD, conformal intervals, SHAP, quantile VaR
│
├── Layer 3 · Financial     layers/layer3_financial/controller.py
│   FinancialLayerController → BOM-weighted COGS, P&L waterfall, scenario shocks
│
├── Layer 4 · Simulation    layers/layer4_simulation/controller.py
│   SimulationLayerController → Monte Carlo 10K (t5 fat-tail), 7 presets,
│                                hedge optimiser, monthly EBIT fan, variance decomposition
│
└── Layer 5 · Governance    layers/layer5_governance/controller.py
    GovernanceLayerController → GICLLMEngine, AuditTrail, BiasTracker, ExplainabilityEngine
```

**Frontend:** React SPA · 11 pages · Recharts · Tailwind · Vercel-deployable  
**Backend:** FastAPI · 33 REST routes · Pydantic v2 · CORS · OpenAPI docs at `/docs`  
**Database:** Supabase PostgreSQL (migration scripts in `supabase/`)

---

## Project Structure

```
GIC_Financial_Intelligence/
├── CLAUDE.md                    # AI assistant setup guide
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
│   ├── data/                    # DataLoader, DataRouter, connectors, synthetic generator
│   │   ├── data_loader.py       # _resolve_path(): real first, synthetic fallback
│   │   ├── data_router.py       # Factory: live | parquet | synthetic
│   │   └── connectors/          # yfinance_connector, fred_connector, ccxt_connector
│   ├── models/                  # All ML models
│   │   ├── commodity_forecast.py
│   │   ├── conformal.py         # ACI split-conformal
│   │   ├── explainability_shap.py
│   │   ├── change_point.py      # CUSUM + BOCPD
│   │   ├── quantile_forecast.py # XGBoost quantile VaR
│   │   ├── hedge_optimizer.py
│   │   └── warranty_model.py
│   ├── simulation/
│   │   └── monte_carlo.py       # Student t(5) MC, run_monthly_fan(), decompose_variance()
│   ├── governance/              # Audit trail, bias tracking, explainability
│   └── insights/                # InsightEngine, variance bridge, EWS
├── auth/                        # RBAC — models, permissions, security, store
├── api/
│   └── index.py                 # Vercel ASGI entry point
├── frontend/                    # React SPA
│   ├── src/
│   │   ├── pages/               # 11 pages
│   │   ├── components/          # Charts, Layout, Insights, Realtime, Common
│   │   ├── auth/                # AuthContext, ProtectedRoute, PermissionGate
│   │   ├── context/             # RealtimeContext (singleton WebSocket)
│   │   ├── hooks/               # useRealtime (WS + client simulator fallback)
│   │   └── api/                 # client.js — typed API methods
│   └── vercel.json              # SPA rewrite rules
├── scripts/
│   ├── fetch_data.py            # Pull Yahoo Finance + FRED + CCXT → data/raw/
│   ├── generate_data.py         # Generate synthetic data → data/synthetic/
│   └── train_models.py          # Train SARIMAX+XGBoost → models/saved/
├── supabase/
│   ├── migrations/001_init.sql  # Full schema — 5 tables, RLS, 13 indexes
│   └── seed.sql                 # Demo users + sample data
├── data/
│   ├── raw/                     # Real data from fetch_data.py (gitignored)
│   ├── external/                # Parquet format (gitignored)
│   └── synthetic/               # Generated CSV fallback
├── config/
│   └── settings.yaml            # All config: data sources, commodities, financial params
├── docs/                        # 15 documentation files
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
# Set REACT_APP_API_URL=https://your-backend.onrender.com
```

### Backend → Render (recommended) or Vercel Serverless
```bash
# Render: connect repo, set start command:
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT

# Vercel serverless: root vercel.json already configured
vercel deploy --prod
```

### Database → Supabase
```bash
# In Supabase SQL Editor:
-- Run supabase/migrations/001_init.sql
-- Run supabase/seed.sql
```

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for full step-by-step with env vars.

---

## API Reference

### Core Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/auth/login` | — | Login → JWT token |
| `GET` | `/auth/demo-profiles` | — | Quick-login profiles |
| `GET` | `/pnl/annual` | User | Annual KPI strip (revenue, EBIT, gross margin) |
| `POST` | `/pnl/shock` | User | BOM-weighted commodity shock P&L waterfall |
| `GET` | `/pnl/regime` | User | Hurst regime for all 12 commodities |
| `POST` | `/forecast/commodity` | User | SARIMAX+XGBoost single commodity forecast |
| `GET` | `/forecast/commodity-index` | User | BOM-weighted composite commodity index |
| `GET` | `/forecast/elasticity` | User | Price elasticity per segment |
| `POST` | `/simulation/scenario` | Admin | Monte Carlo scenario (10K sims, fat-tail t5) |
| `GET` | `/simulation/compare-presets` | User | 7-scenario comparison table |
| `GET` | `/simulation/variance-decomposition` | User | MC variance attribution (commodity/demand/FX %) |
| `GET` | `/simulation/monthly-fan` | User | 12-month EBIT fan chart (p5/p25/mean/p75/p95) |
| `GET` | `/insights/feed` | User | Ranked InsightCards with £ quantification |
| `GET` | `/insights/variance-bridge` | User | Plan-to-Perform EBIT waterfall |
| `GET` | `/insights/early-warning` | User | Risk score 0–100 + top drivers |
| `GET` | `/insights/warranty/summary` | User | Warranty failure forecast |
| `GET` | `/intelligence/change-points/{commodity}` | Admin | CUSUM+BOCPD regime alerts |
| `GET` | `/intelligence/quantile-var` | Admin | Asymmetric 5th/95th VaR bands |
| `GET` | `/realtime/snapshot` | — | Current market snapshot |
| `WS` | `/ws/market` | — | Live market tape (2s ticks) |

Full OpenAPI spec: `http://localhost:8000/docs`

---

## Documentation

| File | Contents |
|------|---------|
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | **What data is used where, how it's fetched, and why** |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Installation, first run, common tasks |
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | Layer-by-layer design, module reference |
| [docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md) | Algorithm details, API table, RBAC matrix |
| [docs/WHY_HOW_IMPACT.md](docs/WHY_HOW_IMPACT.md) | Every feature: What / Why / How / Impact |
| [docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md) | ROI model, pricing, target customer profile |
| [docs/SELLING_DECK.md](docs/SELLING_DECK.md) | Evidence-based pitch, objection handling, demo script |
| [docs/COMPETITIVE_ANALYSIS.md](docs/COMPETITIVE_ANALYSIS.md) | vs Anaplan / Pigment / o9 / Kinaxis / SAP IBP |
| [docs/ROADMAP.md](docs/ROADMAP.md) | P0/P1/P2 priorities, feature backlog |
| [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) | 10-dimension scorecard vs SOTA and competitors |
| [docs/RESEARCH_WOWFACTORS.md](docs/RESEARCH_WOWFACTORS.md) | SOTA survey — N-BEATS, TFT, TimesFM, BOCPD |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Vercel + Supabase deployment guide |
| [docs/OUTPUT_GUIDE.md](docs/OUTPUT_GUIDE.md) | How to read reports and dashboard pages |
| [CLAUDE.md](CLAUDE.md) | AI assistant setup and architecture guide |

---

## Frontend Pages

| Page | Route | Access | Live Data Source |
|------|-------|--------|-----------------|
| Landing | `/` | Public | — |
| Login | `/login` | Public | — |
| Executive Summary | `/app/executive` | User | `/insights/early-warning` + realtime WS |
| Commodity Intelligence | `/app/commodity` | User | `/forecast/commodity` (SARIMAX+XGBoost) |
| Financial P&L | `/app/pnl` | User | `/pnl/annual` (KPI strip); waterfall static |
| Scenario Simulation | `/app/simulation` | Admin/User | `/simulation/scenario` + fan + variance decomp |
| Market Monitor | `/app/market` | User | `/ws/market` WebSocket (commodity & FX live) |
| Insights Centre | `/app/insights` | User | `/insights/feed` |
| Variance Bridge | `/app/variance` | User | `/insights/variance-bridge` |
| Warranty Analytics | `/app/warranty` | User | `/insights/warranty/summary` |
| Governance | `/app/governance` | Admin/User | `/intelligence/*` endpoints |
| Data Explorer | `/app/data` | Admin | `/forecast/commodity-index` + admin endpoints |

---

## RBAC Permissions

Admin has all 20 permissions. User has 9 read/sandbox permissions.

Key Admin-only: `RUN_SIMULATION`, `MANAGE_MODELS`, `TRIGGER_RETRAINING`, `TRIGGER_DATA_FETCH`, `VIEW_AUDIT_FULL`, `EXPORT_REPORTS`, `EDIT_SCENARIOS`, `MANAGE_THRESHOLDS`

Key User: `VIEW_DASHBOARD`, `VIEW_EXECUTIVE_SUMMARY`, `VIEW_FORECASTS`, `VIEW_INSIGHTS`, `VIEW_AGGREGATED_DATA`, `VIEW_MARKET_MONITOR`, `VIEW_AUDIT_SUMMARY`, `VIEW_WARRANTY`, `RUN_SANDBOX_SIMULATION`

---

## Branch

Active development branch: **`dev`**

```bash
git checkout dev
```
