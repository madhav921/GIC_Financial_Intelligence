# Architecture Deep Dive — GIC Financial Intelligence Platform

This guide explains the 5-layer architecture, the role of each component, and how data flows end-to-end.

---

## System Overview

```
orchestrator.py — GICOrchestrator.run_full_pipeline()
│
├── Layer 1 · Data          layers/layer1_data/controller.py
│   DataLayerController → O-U synthetic / CSV / Parquet → commodity, macro, sales, BOM, warranty
│
├── Layer 2 · Intelligence  layers/layer2_intelligence/controller.py
│   IntelligenceLayerController → SARIMAX+XGBoost ensemble, Hurst regime,
│                                  CUSUM+BOCPD change-point (G7), conformal ACI (G5),
│                                  TreeSHAP explainability (G6), quantile VaR (G11)
│
├── Layer 3 · Financial     layers/layer3_financial/controller.py
│   FinancialLayerController → BOM-weighted COGS, P&L waterfall, segment revenue
│
├── Layer 4 · Simulation    layers/layer4_simulation/controller.py
│   SimulationLayerController → Monte Carlo 10K, t(df=5) fat tails, 7 presets,
│                                hedge optimiser (portfolio theory), fan chart
│
└── Layer 5 · Governance    layers/layer5_governance/controller.py
    GovernanceLayerController → GICLLMEngine, AuditTrail, BiasTracker, ExplainabilityEngine

API:   FastAPI (src/api/app.py) — 31 REST routes + 1 WebSocket
UI:    React SPA (frontend/) — 11 pages, Recharts, Tailwind
Auth:  RBAC (auth/) — 20-permission matrix, HMAC-SHA256 JWT
DB:    Supabase PostgreSQL (supabase/) — 5 tables, RLS
Deploy: Vercel (api/index.py entry point)
```

---

## Layer Architecture

### Layer 1 — Data Ingestion

```
Input Sources                 Pipeline                 Storage
─────────────                 ────────                 ───────
• Yahoo Finance (market)  ┐
• FRED (macro)            │   ┌──────────────────┐    ┌────────────────────┐
• CCXT (crypto stubs)     ├──▶│ DataLayerController│──▶│ data/synthetic/    │
• SAP/SF (future stubs)   │   │ (Ornstein-Uhlenbeck│   │ (CSV — commodity,  │
• Synthetic O-U generator │   │  synthetic gen)    │   │  macro, sales, BOM,│
                          ┘   └──────────────────┘    │  warranty)         │
                                                       └────────────────────┘

Result: Single abstraction — swap data sources in one config line.
        All downstream code is source-agnostic.
```

**Key files:**
- `layers/layer1_data/controller.py` — `DataLayerController.load_all()`, `generate_synthetic_data()`
- `src/data/synthetic_generator.py` — O-U processes with realistic commodity correlations

---

### Layer 2 — Predictive Intelligence

```
For each of 12 commodities:

  Historical prices  ┐
  + Macro context    ├─▶ Hurst exponent ─▶ Regime: Trending / Mean-reverting
  + Regime signal    ┘

  ┌─ Adaptive Ensemble ──────────────────────────────────────────────┐
  │  • XGBoost (G1)     — if Trending       — weight 60%            │
  │  • SARIMAX          — if Mean-reverting  — weight 25%            │
  │  • Futures curve    — if liquid          — weight 15%            │
  └──────────────────────────────────────────────────────────────────┘
           │
           ├─▶ ConformalForecaster (G5) — ACI split-conformal ≥90% coverage
           ├─▶ ShapExplainer (G6)       — TreeSHAP per-commodity feature drivers
           ├─▶ ChangePointDetector (G7) — CUSUM + BOCPD regime shift alerts
           └─▶ QuantileForecaster (G11) — XGBoost 2.x joint quantile VaR
```

**Key files:**
- `layers/layer2_intelligence/controller.py` — `IntelligenceLayerController`
- `src/models/commodity_forecast.py` — SARIMAX + XGBoost ensemble
- `src/models/conformal.py` — ACI split-conformal prediction intervals
- `src/models/explainability_shap.py` — TreeSHAP attribution
- `src/models/change_point.py` — CUSUM + BOCPD (G7)
- `src/models/quantile_forecast.py` — XGBoost 2.x joint quantile (G11)
- `src/models/regime_detector.py` — Hurst exponent regime classification

---

### Layer 3 — Financial P&L

```
Input: Commodity forecasts + demand forecasts
────────────────────────────────────────────

Revenue Side
  Base Revenue = £176B
  Elasticity-adjusted Revenue = Base × (Forecast_Demand_t / Base_Demand_0)^elasticity
  Segments: EV 25% | Luxury SUV 35% | Performance 20% | Premium SUV 20%

COGS Side
  Material Spend = Σ (BOM_Weight_i × Quantity_Produced × Price_i)
  BOM_Weights: Steel 22% | Lithium 18% | Aluminum 12% | ... (12 total, sum=1.0)
  COGS = Material_Spend + Manufacturing + Labor + Overhead

EBIT
  Gross Margin = Revenue − COGS
  Fixed Costs: Warranty 1.8% | Depreciation 2.2%
  EBIT = Gross Margin − Fixed Costs − R&D − SG&A

Output: Monthly P&L | Bear/Base/Bull scenarios | EBIT waterfall by driver
```

**Key files:**
- `layers/layer3_financial/controller.py` — `FinancialLayerController`
- `src/insights/variance_bridge.py` — Plan-to-Perform EBIT waterfall decomposition

---

### Layer 4 — Simulation & Risk

```
Monte Carlo Engine (10,000 simulations)
───────────────────────────────────────
  For each sim path:
  1. Sample commodity returns — multivariate t(df=5) fat-tail distribution
  2. Simulate demand shocks (EV adoption, macro, FX)
  3. Propagate through financial model → P&L distribution
  4. Compute VaR(95%), CVaR(95%), margin distribution

  Result: VaR(95%) = £19.3bn | CVaR(95%) = £8.3bn
          Risk split: Commodity 66% / FX 26% / Demand 8%

Quantile VaR (G11 — XGBoost 2.x joint quantile)
──────────────────────────────────────────────────
  Lag features on commodity index → joint fit at τ={0.05,0.25,0.50,0.75,0.95}
  Output: var_5pct=83.22 | median=84.07 | var_95pct=87.94

Hedge Optimiser (portfolio theory)
───────────────────────────────────
  Minimize EBIT variance subject to hedge cost + liquidity constraints
  Output: Optimal hedge ratio per commodity | expected savings vs. naive %
```

**Key files:**
- `layers/layer4_simulation/controller.py` — `SimulationLayerController`
- `src/simulation/monte_carlo.py` — Monte Carlo with t(df=5) fat tails
- `src/models/hedge_optimizer.py` — Portfolio-theory hedge optimisation
- `src/models/quantile_forecast.py` — G11 quantile VaR

---

### Layer 5 — Governance & Explainability

```
GICLLMEngine
  Cascade: Ollama → HuggingFace flan-t5 → deterministic template
  Generates: executive narrative, commodity outlook, risk summary
  Swap to Claude API: set ANTHROPIC_API_KEY + 20-line change

AuditTrail
  Immutable JSONL at data/audit/audit_trail.jsonl
  20 events per run: model_trained, forecast_generated, scenario_run, ...
  Stored to Supabase audit_events table in production

BiasTracker
  Tracks directional accuracy, overestimation/underestimation patterns
  Flags systematic bias for re-calibration

ExplainabilityEngine
  TreeSHAP feature importances → LLM narrative
  "Steel +15% drove +£320M COGS impact via BOM weight 22%"
```

**Key files:**
- `layers/layer5_governance/controller.py` — `GovernanceLayerController`
- `src/governance/audit_trail.py` — Append-only JSONL audit
- `src/governance/bias_tracker.py` — Directional bias analysis
- `src/governance/explainability_engine.py` — SHAP → narrative

---

## API Layer

**FastAPI app** (`src/api/app.py`) — 31 REST routes + 1 WebSocket

```
Route Groups            Module
────────────────────    ─────────────────────────────────
GET  /health            src/api/routes/health.py
POST /auth/login        src/api/routes/auth.py
GET  /auth/me           src/api/routes/auth.py
POST /forecast/commodity         src/api/routes/forecast.py
GET  /forecast/commodity-index   src/api/routes/forecast.py
GET  /forecast/elasticity        src/api/routes/forecast.py
POST /pnl/build                  src/api/routes/pnl.py
GET  /pnl/annual                 src/api/routes/pnl.py
POST /pnl/shock                  src/api/routes/pnl.py
GET  /pnl/regime                 src/api/routes/pnl.py
POST /simulation/scenario        src/api/routes/simulation.py
GET  /simulation/presets         src/api/routes/simulation.py
GET  /simulation/compare-presets src/api/routes/simulation.py
GET  /insights/feed              src/api/routes/insights.py
GET  /insights/variance-bridge   src/api/routes/insights.py
GET  /insights/early-warning     src/api/routes/insights.py
GET  /insights/warranty/summary  src/api/routes/insights.py
GET  /intelligence/change-points/{commodity}  src/api/routes/intelligence.py
GET  /intelligence/change-points              src/api/routes/intelligence.py
GET  /intelligence/quantile-var               src/api/routes/intelligence.py
GET  /realtime/snapshot          src/api/routes/realtime.py
WS   /ws/market                  src/api/routes/realtime.py
```

**Vercel entry point:** `api/index.py` → `from src.api.app import app`

---

## Frontend Layer

**React SPA** (`frontend/src/`) — 11 pages, Recharts, Tailwind

```
frontend/src/
├── pages/               11 pages
│   ├── Landing.jsx       public — value prop, stat band, feature grid
│   ├── Login.jsx         public — quick-demo buttons
│   ├── ExecutiveSummary.jsx   /app/executive — KPI strip, risk gauge, insights
│   ├── CommodityIntelligence.jsx /app/commodity — forecast chart, SHAP, heatmap
│   ├── FinancialPnL.jsx       /app/pnl — EBIT waterfall, sensitivity slider
│   ├── ScenarioSimulation.jsx /app/simulation — MC histogram, tornado, VaR
│   ├── MarketMonitor.jsx      /app/market — live tape, sparklines, FX panel
│   ├── InsightsCentre.jsx     /app/insights — ranked InsightCards
│   ├── VarianceBridge.jsx     /app/variance — Plan-to-Perform waterfall
│   ├── WarrantyAnalytics.jsx  /app/warranty — failure modes, accrual adequacy
│   ├── Governance.jsx         /app/governance — audit trail, bias, narratives
│   └── DataExplorer.jsx       /app/data (Admin only) — raw preview, schema
│
├── components/
│   ├── Layout/           Sidebar.jsx, Header.jsx (live indicator, clock, user)
│   ├── Charts/           QuantileChart, EarlyWarningGauge, RiskDecomposition,
│   │                     CorrelationHeatmap, FanChart
│   ├── realtime/         LiveMarketTape.jsx, LiveKpiStrip.jsx
│   └── Insights/         InsightCard.jsx, WarrantyCard.jsx
│
├── context/
│   └── RealtimeContext.jsx   Singleton WebSocket provider (prevents triple-instantiation)
│
├── auth/
│   ├── AuthContext.jsx        JWT login/logout, user state
│   ├── ProtectedRoute.jsx     Redirect to /login if unauthenticated
│   └── PermissionGate.jsx     Render/hide based on user permissions
│
├── hooks/
│   └── useRealtime.js         WebSocket hook — mean-reverting feed + JS simulator fallback
│
└── api/
    └── client.js              Axios client — 19 typed API methods + WS URL helper
```

**Key architectural pattern:** `RealtimeContext` singleton wraps the app shell so a single WebSocket connection is shared across Header, LiveMarketTape, LiveKpiStrip, and ExecutiveSummary.

---

## Auth Layer

**RBAC** (`auth/`) — 20-permission matrix

```
auth/
├── models.py        User, Role, Permission dataclasses
├── permissions.py   ROLE_PERMISSIONS dict — Admin (20 perms) / User (9 perms)
├── security.py      pbkdf2_hmac password hashing, HMAC-SHA256 JWT
├── store.py         users.json store — auto-seeded with admin + user
└── dependencies.py  FastAPI Depends() helpers — require_permission()

Key Admin-only:  RUN_SIMULATION, MANAGE_MODELS, TRIGGER_RETRAINING,
                 VIEW_AUDIT_FULL, EXPORT_REPORTS, EDIT_SCENARIOS
Key User:        VIEW_DASHBOARD, VIEW_FORECASTS, RUN_SANDBOX_SIMULATION,
                 VIEW_AGGREGATED_DATA, VIEW_AUDIT_SUMMARY
```

---

## Database Layer

**Supabase PostgreSQL** (`supabase/`)

```
supabase/
├── migrations/001_init.sql   5 tables + RLS + 13 indexes (idempotent)
│   ├── users                 id, username, hashed_password, role, permissions
│   ├── audit_events          id, event_type, user_id, payload, timestamp
│   ├── forecasts             id, commodity, horizon, predictions, created_at
│   ├── scenarios             id, name, params, results, created_at
│   └── bias_alerts           id, commodity, metric, value, threshold, created_at
│
└── seed.sql                  Demo users (admin/admin123, user/user123 via pgcrypto)
                              + 5 sample audit events + 3 commodity forecasts
```

---

## Data Flow: Prices → P&L → Dashboard

```
Step 1  Data Ingestion
        DataLayerController.load_all()
        └── Generates synthetic O-U prices, macro indicators, sales volumes, BOM

Step 2  Forecasting
        IntelligenceLayerController.train_and_forecast()
        ├── Hurst regime detection per commodity
        ├── SARIMAX + XGBoost adaptive ensemble → 12-month forecasts
        ├── ConformalForecaster → ≥90% coverage prediction intervals
        ├── ChangePointDetector → regime shift alerts (auto-reforecast if confidence >0.6)
        └── QuantileForecaster → asymmetric 5th/95th VaR bands

Step 3  Financial P&L
        FinancialLayerController.build_pnl()
        └── BOM-weighted COGS + revenue → EBIT waterfall

Step 4  Risk Simulation
        SimulationLayerController.run_monte_carlo()
        └── 10K paths, t(df=5) fat tail → VaR(95%)=£19.3bn, CVaR=£8.3bn

Step 5  Governance
        GovernanceLayerController.generate_narratives()
        └── LLM (flan-t5 / Ollama / template) → executive insight
        └── AuditTrail → 20 events logged to data/audit/audit_trail.jsonl

Step 6  API → Frontend
        FastAPI /pnl/annual, /forecast/commodity, /simulation/compare-presets, ...
        └── React SPA fetches via client.js → Recharts visualisations
        └── WebSocket /ws/market → live commodity tape + EBIT nowcast
```

---

## Module Reference

### `layers/` — Layer Controllers

| Controller | Responsibility |
|-----------|---------------|
| `layer1_data/controller.py` | Data loading, synthetic generation, schema validation |
| `layer2_intelligence/controller.py` | Forecasting, regime detection, SOTA modules (G5/G6/G7/G11) |
| `layer3_financial/controller.py` | P&L build, BOM weighting, annual summary |
| `layer4_simulation/controller.py` | Monte Carlo, scenario comparison, hedge optimiser |
| `layer5_governance/controller.py` | LLM narratives, audit logging, bias tracking |

---

### `src/models/` — ML Models

| File | Purpose |
|------|---------|
| `commodity_forecast.py` | SARIMAX + XGBoost ensemble, 5-fold CV, regime weighting |
| `conformal.py` | ACI split-conformal prediction intervals (≥90% coverage) |
| `explainability_shap.py` | TreeSHAP feature attribution per commodity |
| `change_point.py` | CUSUM + BOCPD change-point detection (G7) |
| `quantile_forecast.py` | XGBoost 2.x joint quantile VaR/CVaR (G11) |
| `regime_detector.py` | Hurst exponent → Trending / Mean-reverting / Volatile |
| `hedge_optimizer.py` | Portfolio-theory optimal hedge ratio |
| `warranty_model.py` | Weibull failure modes, EV learning curve, accrual adequacy |
| `backtesting.py` | Walk-forward backtesting, coverage calibration |

---

### `src/api/` — FastAPI Application

| File | Purpose |
|------|---------|
| `app.py` | App factory — CORS, router registration, startup events |
| `routes/auth.py` | Login, /me, demo-profiles, permissions |
| `routes/forecast.py` | Commodity forecast, index, elasticity |
| `routes/pnl.py` | Annual P&L, shock waterfall, regime |
| `routes/simulation.py` | Scenario run, presets, compare-presets |
| `routes/insights.py` | Feed, variance bridge, early warning, warranty |
| `routes/intelligence.py` | Change-point alerts, quantile VaR (G7/G11) |
| `routes/realtime.py` | Market snapshot, WebSocket /ws/market |
| `routes/health.py` | /health liveness check |

---

### `src/insights/` — Intelligence Engine

| File | Purpose |
|------|---------|
| `insight_engine.py` | Ranked InsightCards with £-quantified recommended actions |
| `variance_bridge.py` | Plan-to-Perform EBIT waterfall decomposition by driver |
| `early_warning.py` | Risk score 0–100, EWS signal aggregation |
| `recommendations.py` | Hedge and procurement action recommendations |

---

### `src/governance/` — Explainability & Audit

| File | Purpose |
|------|---------|
| `audit_trail.py` | Append-only JSONL audit trail |
| `bias_tracker.py` | Directional accuracy, systematic bias detection |
| `explainability_engine.py` | SHAP → narrative, model selection explanation |
| `llm_engine.py` | Ollama → flan-t5 → template cascade |

---

### `src/simulation/` — Risk Engine

| File | Purpose |
|------|---------|
| `monte_carlo.py` | 10K simulation, t(df=5) fat tails, correlation matrix |
| `scenario_engine.py` | 7 preset scenarios (Bear/Base/Bull/commodity shocks) |

---

## Configuration

**File:** `config/settings.yaml`

```yaml
# BOM weights (12 commodities, sum=1.0)
bom_weights:
  Steel: 0.22
  Lithium: 0.18
  Aluminum: 0.12
  Cobalt: 0.07
  # ... 8 more

# Forecast horizon
forecast_horizon_months: 12

# Monte Carlo
monte_carlo_runs: 10000
fat_tail_df: 5          # Student-t degrees of freedom

# 7 scenarios
scenarios:
  - {name: base, lithium: 1.0, aluminum: 1.0, demand: 1.0}
  - {name: bear, lithium: 1.2, aluminum: 1.15, demand: 0.95}
  - {name: bull, lithium: 0.8, aluminum: 0.85, demand: 1.05}
  # ... 4 more
```

---

## Testing & Validation

```bash
pytest tests/ -v
```

What's tested:
- Data pipeline correctness (synthetic generation, schema)
- Model accuracy — walk-forward CV MAPE per commodity
- Conformal coverage — ACI calibration ≥90% hold-out coverage
- Financial model math — COGS, margin calculations, BOM-weight sum
- Monte Carlo calibration — CI coverage, fat-tail shape
- API contract — response schemas, HTTP verbs, auth enforcement

---

## Production Deployment

**Current state:** Vercel (backend + frontend) + Supabase PostgreSQL

```
Backend  → api/index.py (ASGI entry) + vercel.json (root)
Frontend → frontend/vercel.json (SPA rewrite rules)
Database → supabase/migrations/001_init.sql + supabase/seed.sql
```

**Production readiness roadmap:**
1. **Data layer** — Replace `synthetic_generator.py` with SAP/Oracle connector
2. **Model layer** — Retrain daily on live commodity feed; warehouse actuals
3. **Financial layer** — Wire general ledger COGS actuals → variance tracking
4. **Simulation** — Risk model sign-off by CFO + Quantitative Risk team
5. **Governance** — Audit trail feeds into quarterly IFRS 9 financial review

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `Module not found` | Missing dependency | `pip install -r requirements.txt` in repo root |
| Frontend shows no data | Backend not running | `uvicorn src.api.app:app --port 8000` first |
| WebSocket shows "Simulated" | Backend WS unreachable | Expected — JS simulator kicks in automatically |
| JWT auth fails | `auth/users.json` corrupted | Delete `auth/users.json` — auto-regenerated on startup |
| Forecast MAPE high | Regime mis-classified | Inspect `src/models/regime_detector.py` Hurst output |
| Monte Carlo crash | Singular covariance matrix | Verify no duplicate commodity columns in data |
| EBIT doesn't match plan | BOM weights wrong | Confirm `sum(bom_weights.values()) == 1.0` in config |

---

## Further Reading

| Document | Contents |
|----------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, first run, common tasks (5 min) |
| [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md) | Algorithm details, API table, RBAC matrix |
| [WHY_HOW_IMPACT.md](WHY_HOW_IMPACT.md) | Every feature: What / Why / How / Impact |
| [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) | How to interpret every number in the dashboard |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Vercel + Supabase step-by-step deployment |
| [ROADMAP.md](ROADMAP.md) | ✅/⏳/❌ checklist, P0/P1/P2 priorities |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | 10-dimension scorecard vs SOTA and competitors |
| [FULL_ARCHITECTURE_RUN.md](FULL_ARCHITECTURE_RUN.md) | Full pipeline run trace with real metrics |
