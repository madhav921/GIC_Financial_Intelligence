# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GIC Financial Intelligence Platform — an AI-powered Plan-to-Perform engine for automotive OEM commodity risk (JLR-scale). Translates live commodity market signals into quantified EBIT impact, VaR-bounded risk, and hedge recommendations.

## Essential Commands

### Backend
```bash
# Start API server (required for all frontend live data)
uvicorn src.api.app:app --reload --port 8000

# Fetch real-world data (Yahoo Finance + FRED + CCXT) → data/raw/
python scripts/fetch_data.py

# Generate synthetic data fallback → data/synthetic/
python scripts/generate_data.py

# Train all commodity forecast models → models/saved/
python scripts/train_models.py

# Run full 5-layer pipeline (returns full results dict)
python -c "from orchestrator import GICOrchestrator; r = GICOrchestrator().run_full_pipeline(); print(r.keys())"
```

### Frontend
```bash
cd frontend
npm install        # first time
npm start          # dev server on :3000
npm run build      # production build
```

### Tests
```bash
pytest tests/ -v
pytest tests/test_api.py::test_health -v   # single test
```

### Linting
```bash
ruff check src/                # backend lint
ruff check src/ --fix          # auto-fix
cd frontend && npx eslint src/ # frontend lint
```

## Architecture

5-layer pipeline wired by `orchestrator.py → GICOrchestrator.run_full_pipeline()`:

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 · Data | `layers/layer1_data/` | Load commodity/macro/sales/BOM from real or synthetic sources |
| 2 · Intelligence | `layers/layer2_intelligence/` | SARIMAX+XGBoost ensemble, CUSUM+BOCPD regime, SHAP, conformal intervals |
| 3 · Financial | `layers/layer3_financial/` | BOM-weighted COGS, P&L waterfall, scenario shocks |
| 4 · Simulation | `layers/layer4_simulation/` | Monte Carlo 10K sims (fat-tail t5), VaR/CVaR, hedge optimiser |
| 5 · Governance | `layers/layer5_governance/` | LLM narratives, audit trail, bias tracking, explainability |

**API**: `src/api/app.py` (FastAPI) — 8 route modules in `src/api/routes/`  
**Frontend**: `frontend/src/` — React SPA, 11 pages, Tailwind, Recharts  
**Auth**: `auth/` — RBAC with 20-permission matrix, HMAC-SHA256 JWT

## Data Flow (Real → Synthetic Priority)

```
scripts/fetch_data.py           # pulls Yahoo Finance + FRED + CCXT
    → data/raw/commodity_prices.csv
    → data/raw/macro_indicators.csv
    → data/external/*.parquet

src/data/data_loader.py         # _resolve_path() checks data/raw/ FIRST
    → falls back to data/synthetic/ if raw not present

config/settings.yaml            # market_data_source: live (default)
                                 # operational_data_source: synthetic (default)
src/data/data_router.py         # factory: live → YFinanceMarketSource
                                 #          parquet → ParquetMarketSource
                                 #          synthetic → _SyntheticMarketSource
```

**Key rule**: never modify `data/synthetic/` files — always run `fetch_data.py` to populate `data/raw/`.

## Key Configuration

`config/settings.yaml` controls everything:
- `market_data_source: live` — commodity/FX from Yahoo Finance directly at request time
- `operational_data_source: synthetic` — change to `parquet` after dropping real sales data in `data/raw/`
- `commodities[]` — 12 tracked materials with BOM weights, yfinance tickers, model preferences
- `vehicle_segments[]` — JLR-calibrated: Luxury SUV / Premium SUV / Performance / EV

## Frontend Pages & Data Sources

| Page | Route | Data Source |
|------|-------|-------------|
| Executive Summary | `/app/executive` | `/insights/early-warning` + `/insights/feed` (live) |
| Commodity Intelligence | `/app/commodity` | `/forecast/commodity` (SARIMAX+XGBoost, live); history seeded |
| Financial P&L | `/app/pnl` | `/pnl/annual` for KPI strip (live); waterfall static |
| Scenario Simulation | `/app/simulation` | `/simulation/scenario` + `/simulation/monthly-fan` + `/simulation/variance-decomposition` |
| Market Monitor | `/app/market` | WebSocket `/ws/market` (live); indices/macro static reference |
| Insights Centre | `/app/insights` | `/insights/feed` (live) |
| Variance Bridge | `/app/variance` | `/insights/variance-bridge` (live) |
| Warranty Analytics | `/app/warranty` | `/insights/warranty/summary` (live) |
| Governance | `/app/governance` | `/intelligence/*` endpoints (live) |
| Data Explorer | `/app/data` | `/forecast/commodity-index` + admin endpoints |

## Critical Patterns

### API client (`frontend/src/api/client.js`)
All calls go through `gicApi.*`. The client has two-tier fallback:
1. Real backend via axios
2. Client-side mock auth when backend unreachable (for demo)

### Monte Carlo (`src/simulation/monte_carlo.py`)
- Student's t fat-tails, df=5
- `run_monthly_fan(n=2000)` → 12-month EBIT percentile fan
- `decompose_variance(n=3000)` → commodity/demand/FX attribution

### Realtime feed (`src/api/routes/realtime.py`)
- `MarketFeed` seeds prices from `data/synthetic/commodity_prices.csv`, adds mean-reverting walk
- WebSocket `/ws/market` → frontend `useRealtime` hook → `RealtimeContext`
- If WebSocket fails, `useRealtime` automatically starts client-side simulator

### Auth
- `POST /auth/login` → JWT (HMAC-SHA256)
- `auth/permissions.py` → `PERMISSIONS` enum used in `LockedButton` / `PermissionGate`
- Admin-only: `RUN_SIMULATION`, `MANAGE_MODELS`, `TRIGGER_RETRAINING`, `VIEW_AUDIT_FULL`

## Common Pitfalls

- Backend returns P&L values in **USD** (config uses `avg_price_usd`). Frontend divides by 1.27 to display in **GBP**.
- Commodity names differ: frontend uses "Natural Gas" / "ABS Resin"; backend CSV uses "Natural_Gas" / "ABS_Resin". Use `.replace(/ /g, '_')` when calling forecast APIs.
- `top_drivers` from `/insights/early-warning` returns `[{component, contribution, score}]` objects, not strings — always apply type guard before rendering.
- Monte Carlo `simulation_stats` key is returned from `/simulation/scenario`; the UI maps it to internal `stats`.
- `data/raw/` is gitignored — run `fetch_data.py` on each new machine.

## Dependencies (Local Full Stack)

```bash
pip install -e ".[full]"    # backend + all connectors (yfinance, fredapi, ccxt, polars, scipy)
pip install -e ".[dev]"     # + pytest, ruff, mypy
```

Vercel deployment uses slim `pyproject.toml` dependencies (no `[full]`). Real data connectors only work in local/Docker environments.
