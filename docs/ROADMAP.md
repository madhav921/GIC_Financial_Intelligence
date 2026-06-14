# Product Roadmap — GIC Financial Intelligence

*What's built, what's next, what would make this a complete enterprise product.*

---

## Current State Checklist

### Built and Working

- [x] 5-layer architecture (Data / Intelligence / Financial / Simulation / Governance)
- [x] 12-commodity SARIMAX+XGBoost ensemble forecasting with 5-fold CV, 84 months training
- [x] Monte Carlo simulation (10K sims, fat-tail Student-t(df=5), 7 preset scenarios)
- [x] BOM-weighted commodity index and P&L waterfall
- [x] Hurst exponent regime detection (MEAN_REVERTING / TRENDING / VOLATILE, rolling 36-month)
- [x] CUSUM+BOCPD change-point detection with auto-reforecast trigger (G7)
- [x] Split-conformal + ACI prediction intervals with provable coverage (G5)
- [x] SHAP TreeSHAP + permutation fallback (G6)
- [x] Gradient-boosted quantile VaR/CVaR — XGBoost 2.x joint objective (G11)
- [x] Hedge optimiser (portfolio-theory optimal h*, scipy.optimize)
- [x] JSONL immutable audit trail (UUID-keyed, append-only, 20 events per run)
- [x] Bias tracking with >5% alert, >10% escalation
- [x] Open-source LLM cascade (Ollama → HuggingFace flan-t5 → template)
- [x] RBAC auth (20 permissions, Admin/User, HMAC-SHA256 JWT)
- [x] WebSocket real-time market feed (`/ws/market`) with client simulator fallback
- [x] Plan-to-Perform variance bridge (EBIT waterfall)
- [x] Warranty analytics (O-U synthetic, EV learning curve, Weibull)
- [x] InsightCard recommendation engine with £ quantification
- [x] 11-page React dashboard (Recharts, Tailwind)
- [x] Vercel-deployable frontend (vercel.json, SPA rewrites)
- [x] Supabase PostgreSQL migration scripts
- [x] 31-endpoint FastAPI API (31 routes + 1 WebSocket)
- [x] 22/22 Python modules import clean

### Partially Done

- [ ] LLM integration — template fallback works; Ollama/HuggingFace require server running
- [ ] Real market data — O-U synthetic only; yfinance connector exists in `data_router.py` but not wired to live
- [ ] Supabase integration — SQL migrations ready; app still uses JSON file store and in-memory user dict
- [ ] Change-point auto-reforecast wiring — detection module ships; scheduled retrain trigger is pending

### Not Yet Built

- [ ] SAP/Oracle ERP connector (stub only, returns synthetic data)
- [ ] Multi-tenancy (no tenant_id scoping; shared filesystem)
- [ ] Automated retraining pipeline
- [ ] Bloomberg/Refinitiv live feed
- [ ] IFRS 9 hedge accounting documentation generator
- [ ] PDF/Excel export of reports
- [ ] Email/Slack/Teams alert integration
- [ ] Mobile PWA

---

## P0 — Critical for Enterprise GA (~3 months)

| Item | Why Critical | Effort | Impact |
|---|---|---|---|
| Supabase integration | Replace JSON store; production auth; queryable audit | M | HIGH |
| Claude API for LLM | Replace template; board-quality narratives | S | HIGH |
| SAP BAPI connector | Enterprise gate-keeper; no real data = no sign-off | XL | CRITICAL |
| Automated retraining (GitHub Actions) | Model drift; monthly commodity data update cycle | M | HIGH |
| Multi-tenancy (tenant_id in DB) | SaaS mandatory; customer data isolation | L | CRITICAL |
| Data quality monitoring (Great Expectations) | Silent failures produce confident wrong forecasts | M | MEDIUM |
| PDF/Excel export | CFO requirement #1 — hand-off artifact | S | HIGH |
| Email alerts (bias escalation) | IFRS 9 governance; >10% bias needs immediate action | S | MEDIUM |
| WebSocket auth token check | `/ws/market` currently accepts any connection | S | MEDIUM |
| Secret management | `AUTH_SECRET_KEY` hardcoded fallback unsafe in prod | S | HIGH |

---

## P1 — Competitive Differentiation (~6 months)

| Item | Why | Effort |
|---|---|---|
| N-HiTS / TFT forecasting challenger | 2–5% MAPE improvement on volatile tail (NatGas 31%, Palladium 29%) | L |
| Bloomberg/Refinitiv connector | Live data = enterprise credibility; replaces O-U fallback | L |
| yfinance wiring (free tier) | Quick win vs synthetic; already in pyproject.toml | S |
| IFRS 9 documentation generator | Hedge accounting compliance; reads bias tracker + audit trail | M |
| Slack/Teams alert integration | Operational workflow; change-point + bias escalation notifications | S |
| Mobile PWA | CFO field access — Executive Summary + VaR gauge + top InsightCards | M |
| Scenario comparison v2 (fan chart) | CFO presentation quality; overlays multiple scenario distributions | M |
| Rate limiting on API | Security hygiene; prevents abuse without auth | S |

---

## P2 — Wow / Enterprise Upsell (~12 months)

| Item | Why | Notes |
|---|---|---|
| Chronos / TimesFM zero-shot | No per-commodity training; wrap in existing conformal layer | Research-grade — evaluate first |
| Supply chain disruption simulation | Links commodity price to supplier risk via graph propagation | Requires supplier network data |
| Carbon/Scope 3 forecasting | EU ETS carbon price + BOM carbon intensity → EBIT under carbon scenarios | Growing regulatory requirement |
| Weibull warranty insurance premium optimiser | Self-insure vs market insurance vs reinsurance decision with P&L quantification | Unique capability; direct CFO decision |
| Automated RFQ negotiation intelligence | Commodity forecast + supplier quote → fair-value range for procurement playbook | Agent-based; potential ISV partnership |
| Natural language query interface | "What if steel goes up 15%?" → instant scenario in plain English | LLM function-calling; APIs already exist |
| RAG-grounded narrative | LLM reads real financial filings + board minutes for contextualised commentary | Requires document ingestion pipeline |

---

## Immediate Quick Wins (1–2 days each)

These would impress judges or CFOs most at current stage with minimal effort:

1. **Wire yfinance live data** — replace synthetic with real prices (yfinance already in pyproject.toml; `scripts/fetch_data.py` works)
2. **Claude API for LLM narratives** — swap template fallback with Anthropic SDK (1 hour, ~20 lines, one env var `ANTHROPIC_API_KEY`)
3. **PDF executive summary export** — WeasyPrint or ReportLab; huge CFO appeal as hand-off artifact
4. **Email on bias escalation** — `smtplib`, 30 lines; hooks into existing `BiasTracker.compute_bias()` `is_alert=True`
5. **Confidence interval chart overlay** — Recharts `ReferenceArea` on all forecast charts; conformal module already produces the bounds

---

## Technical Debt

| Item | Severity | Fix |
|---|---|---|
| `auth/users.json` → Supabase `users` table | High | Migration SQL ready in `migrations/`; just needs wiring in `auth/store.py` |
| `data/audit/*.jsonl` → `audit_events` table | High | Supabase migration ready; replace `AuditTrail._write_entry()` append with Supabase insert |
| `DataLoader` vs `DataLayerController` | Medium | Two data access patterns in codebase; consolidate to L1 controller interface |
| Test coverage gaps | Medium | Zero unit tests for SOTA modules; add pytest for `CommodityForecastModel` (L2) and `MonteCarloEngine` (L4) as minimum |
| SARIMAX frequency warnings | Low | Set `freq='MS'` explicitly at DatetimeIndex creation to suppress `ValueWarning: No frequency information` |
| CORS `allow_origins=["*"]` | High | Scope to customer domain in production; currently allows any origin |
| No pagination on audit trail | Low | `get_recent_events(limit=100)` reads full JSONL; needs seek-from-end for large files |
| WebSocket no auth | Medium | Add `token` query param check on `/ws/market` connect handshake |
| Model serialisation versioning | Low | XGBoost models saved with `joblib`; add version metadata and schema hash for model registry |
