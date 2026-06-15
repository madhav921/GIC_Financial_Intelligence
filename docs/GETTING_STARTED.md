# Getting Started — GIC Financial Intelligence Platform

Up and running in **one command** (Docker) or **one script** (native).

No external database or API keys required — everything runs self-contained with synthetic data and a local JSON-backed auth store.

---

## What Is GIC?

GIC is a 5-layer AI financial intelligence engine for automotive OEMs. It forecasts commodity prices, quantifies P&L risk, and generates hedge recommendations — in real time, with full explainability and governance.

**Built for:** CFOs, commodity managers, treasury teams, FP&A analysts  
**Core value:** Translates commodity market signals into quantified EBIT impact with provable prediction intervals

---

## Quick Start

### Option A — Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). No Python or Node.js needed.

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

First build takes ~3–5 min. Subsequent `docker compose up` starts in seconds.

### Option B — Native scripts

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

The scripts auto-create the Python venv, install all dependencies, and start both servers. On subsequent runs they skip the install steps and start immediately.

---

## Log In

| Role | Username | Password | Can do |
|------|----------|----------|--------|
| Admin | `admin` | `admin123` | Everything — simulations, audit, exports |
| User | `user` | `user123` | Read-only dashboards, sandbox simulation |

Use the **quick-login buttons** on the Login page for instant demo access.

---

## Run the Full Pipeline (Python)

```python
from orchestrator import GICOrchestrator

engine = GICOrchestrator()
results = engine.run_full_pipeline(n_simulations=10_000)

# Results structure:
# results['layer1_data']         — commodity/macro/sales shapes
# results['layer2_intelligence'] — n_forecasts, commodity_index_latest
# results['layer3_financial']    — total_revenue, ebit, gross_margin
# results['layer4_simulation']   — mc_stats, scenario_comparison, risk_decomposition
# results['layer5_governance']   — narratives, executive_insight, audit_events
# results['pipeline_elapsed_seconds']
```

Expected output: 12 forecasts in ~15 seconds, VaR(95%) ~£19.3bn, 20 audit events.

---

## Dashboard Pages

| Page | URL | What to try |
|------|-----|------------|
| Executive Summary | `/app/executive` | Live commodity tape, risk gauge, top insights |
| Commodity Intelligence | `/app/commodity` | Select a commodity → forecast + change-point alert + SHAP drivers |
| Financial P&L | `/app/pnl` | EBIT waterfall, commodity sensitivity slider |
| Scenario Simulation | `/app/simulation` | Run Monte Carlo → distribution histogram + VaR markers |
| Market Monitor | `/app/market` | Live prices, sparklines, FX panel |
| Insights Centre | `/app/insights` | Ranked InsightCards with £-quantified recommended actions |
| Variance Bridge | `/app/variance` | Plan-to-Perform EBIT waterfall by driver |
| Warranty Analytics | `/app/warranty` | Failure modes, accrual adequacy, cost forecast |
| Governance | `/app/governance` | Audit trail, bias table, LLM narratives (Admin: full detail) |
| Data Explorer | `/app/data` | Admin only — raw data preview, schema, quality |

---

## Key Directories

```
GIC_Financial_Intelligence/
├── orchestrator.py          # Full pipeline entry point
├── layers/                  # 5 layer controllers
├── src/
│   ├── api/                 # FastAPI app + 8 route files
│   ├── models/              # ML models + SOTA (conformal, SHAP, change_point, quantile)
│   ├── simulation/          # Monte Carlo + scenario engine
│   ├── governance/          # Audit trail + bias tracking
│   └── insights/            # InsightEngine, variance bridge, EWS
├── auth/                    # RBAC (models, permissions, security, JWT, store)
├── frontend/src/            # React SPA (11 pages, Recharts, Tailwind)
├── supabase/                # PostgreSQL migration + seed SQL
├── data/synthetic/          # Generated datasets (CSV)
├── data/audit/              # Append-only JSONL audit trail
└── docs/                    # 14 documentation files
```

---

## Common Tasks

### Check change-point detection on Copper
```bash
curl http://localhost:8000/intelligence/change-points/Copper
# → {shifted, n_breaks, last_break_date, confidence, reforecast_recommended}
```

### Get asymmetric quantile VaR
```bash
curl http://localhost:8000/intelligence/quantile-var
# → {var_5pct: 83.22, median_forecast: 84.07, var_95pct: 87.94}
```

### Quick P&L with commodity shock
```python
from orchestrator import GICOrchestrator
result = GICOrchestrator().quick_pnl(commodity_shock=0.10)  # +10% commodities
print(result)  # → {total_revenue, gross_margin, ebit, demand_shock, commodity_shock}
```

### Generate synthetic data
```python
from layers.layer1_data.controller import DataLayerController
datasets = DataLayerController().generate_synthetic_data()
# → commodity_prices, macro_indicators, sales_data, production_inventory, bom_data
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Module not found` | `pip install -r requirements.txt` in repo root (not frontend/) |
| Dashboard shows no data | Start backend first; wait for the health check to pass |
| WebSocket shows "Simulated" | Expected — client simulator runs when backend WS is unreachable |
| Auth fails | Delete `auth/users.json` to regenerate seed users on next start |
| Docker: port already in use | Stop any process using ports 3000 or 8000, then re-run |
| Docker: frontend not loading | Backend may still be initialising; wait ~30s and refresh |
| Windows script blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then retry |

---

## Next Steps

1. **Architecture**: [docs/ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) — layer-by-layer design
2. **Feature detail**: [docs/WHY_HOW_IMPACT.md](WHY_HOW_IMPACT.md) — every module explained
3. **Deploy**: [docs/DEPLOYMENT.md](DEPLOYMENT.md) — Vercel + Supabase
4. **Business case**: [docs/BUSINESS_CASE.md](BUSINESS_CASE.md) — ROI model
5. **Competitor comparison**: [docs/COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)
