# Roadmap — GIC Financial Intelligence Platform

*What's built, what's next, what would make this a complete enterprise product.*

---

## 1. Current State Checklist

### Core Platform
- ✅ 5-layer FastAPI architecture (Data → Intelligence → Financial → Simulation → Governance)
- ✅ 30 REST endpoints + 1 WebSocket (`/ws/market`)
- ✅ React SPA frontend (11 pages) with Recharts + Tailwind
- ✅ RBAC with 20-permission matrix (Admin/User)
- ✅ HMAC-SHA256 JWT, pbkdf2_hmac passwords (200K iterations)
- ✅ OpenAPI schema auto-generated

### Data Layer (L1)
- ✅ Ornstein-Uhlenbeck synthetic generator (12 commodities, 12 macro indicators, 84 months)
- ✅ Polars pipeline with Parquet caching
- ✅ Protocol abstraction (swap data source in 1 config line)
- ✅ Yahoo Finance, FRED, CCXT connectors
- ⏳ SAP BAPI connector (stub only — no live ERP data)
- ❌ Snowflake / Databricks data lake connector
- ❌ Bloomberg / Refinitiv live feed (mean-reverting fallback in place)
- ❌ Great Expectations data quality checks

### Intelligence Layer (L2)
- ✅ SARIMAX (1,1,1)×(1,1,1,12) with macro exogenous
- ✅ XGBoost 2.x ensemble with 60+ features, 5-fold CV
- ✅ Hurst exponent regime detector (MEAN_REVERTING / TRENDING / VOLATILE)
- ✅ Regime-adaptive ensemble blending
- ✅ Split-conformal + ACI prediction intervals
- ✅ CUSUM + BOCPD change-point detection (module shipped)
- ✅ Gradient-boosted quantile regression (XGBoost 2.x joint objective)
- ✅ SHAP TreeExplainer with gain-importance fallback
- ⏳ BOCPD break events → auto-reforecast trigger (module ships; wiring pending)
- ⏳ Conformalized Quantile Regression (CQR) into VaR path
- ❌ Chronos / TimesFM foundation model challenger
- ❌ N-HiTS / TiDE deep learning challenger
- ❌ Automated retraining scheduler

### Financial Layer (L3)
- ✅ BOM-weighted COGS engine (12 commodities, configurable weights)
- ✅ Revenue-side elasticity (4 segments: EV, Luxury SUV, Performance, Premium)
- ✅ EBIT waterfall with warranty + depreciation
- ✅ Plan-to-Perform variance bridge (Volume→Price/Mix→Commodity→FX→Warranty→Other)
- ✅ 7 preset scenario shocks
- ⏳ Regional demand splits (NA/EU/APAC/China) — single region now
- ❌ Real GL actuals integration

### Simulation Layer (L4)
- ✅ Monte Carlo 10,000 simulations, fat-tail t(df=5) commodity shocks
- ✅ VaR(95%) + CVaR(95%) computation
- ✅ Risk decomposition (commodity/FX/demand)
- ✅ Hedge optimiser (portfolio-theory h*, scipy.optimize)
- ✅ 7 scenario comparison table
- ⏳ CVaR-objective hedging (VaR-blend in place; CVaR objective pending)
- ❌ Multi-commodity correlated shock factor model

### Governance Layer (L5)
- ✅ Immutable JSONL audit trail (UUID-keyed, append-only)
- ✅ Bias tracker (mean bias %, direction, trend, >5% alert)
- ✅ SHAP explainability engine
- ✅ GICLLMEngine cascade (Ollama → HuggingFace flan-t5 → template)
- ✅ InsightEngine with prescriptive InsightCards
- ✅ Warranty analytics (EV learning curve + warranty cost model)
- ⏳ LLM function-calling what-if (API endpoints exist; LLM wiring pending)
- ❌ RAG-grounded narrative (LLM reads real financial filings)
- ❌ IFRS 9 hedge documentation auto-generator
- ❌ Email / Slack alert on bias escalation

---

## 2. P0 — Critical for Enterprise GA (~3 months)

These items block production deployment for any paying enterprise customer.

### Real Database (Supabase / PostgreSQL)

**Current:** User store is in-memory dict; audit trail is flat file JSONL.
**Required:** PostgreSQL with proper schemas for users, audit_events, forecast_runs, model_versions.
**Why:** Multi-user concurrency, persistent sessions, query-able audit history, backup/recovery.
**Effort:** M (2–3 weeks)

### ERP Data Connector (SAP BAPI / REST)

**Current:** `src/data/connectors/erp_connector.py` is a stub returning synthetic data.
**Required:** SAP RFC BAPI calls for BOM data, production volumes, GL actuals (MM60/CO04 equivalents).
**Why:** Without real operational data, the financial model uses calibrated approximations. Enterprise customers will not sign off on synthetic data.
**Effort:** L (6–8 weeks including SAP security review)

### Multi-Tenancy (Tenant Isolation)

**Current:** Single-tenant; all data in shared filesystem paths.
**Required:** Tenant ID scoped to all data reads/writes; isolated audit trails; per-tenant model registries.
**Why:** Mandatory for SaaS — customer A must never see customer B's data.
**Effort:** L (4–6 weeks)

### Production-Grade LLM (Claude API / GPT-4)

**Current:** Template fallback is the de facto output in most deployments (Ollama requires local install; HuggingFace flan-t5 produces generic text).
**Required:** Claude API or GPT-4 via function calling for IFRS 9 narrative generation, bias alerts, and board report summaries.
**Why:** Template output is recognisably mechanical to any CFO. Production-quality narrative is the "last mile" of the governance story.
**Effort:** S (3–5 days — API is already wired in LLM engine; replace template with API call)

### Automated Model Retraining Pipeline

**Current:** Manual — run `scripts/run_commodity_pipeline.py`.
**Required:** Scheduled trigger (weekly or on new data arrival) + drift detection → auto-refit → model registry update.
**Tech:** GitHub Actions (simple), Airflow (robust), or Prefect.
**Effort:** M (2–3 weeks)

### Data Quality Monitoring (Great Expectations)

**Current:** No validation on incoming CSV/API data.
**Required:** Schema validation, null checks, range checks (e.g., Lithium price never negative), freshness alerts.
**Why:** Silent data quality failures produce confidently wrong forecasts — the worst outcome.
**Effort:** S (1 week)

---

## 3. P1 — Competitive Differentiation (~6 months)

Moves GIC from parity-with-competitors to clear technical leadership.

### N-HiTS / Temporal Fusion Transformer for Improved Accuracy

- Challenger model trained alongside existing ensemble in walk-forward harness
- Per-commodity winner promoted (or blended) based on CV MAPE
- Target: close NatGas 31% / Palladium 29% MAPE gap by 30–50%
- Source: `src/models/backtesting.py` already provides the comparison harness

### Live Market Data Feeds (Bloomberg / Refinitiv Connector)

- Replace mean-reverting WebSocket fallback with real LME/CME tick data
- Bloomberg B-PIPE or Refinitiv Eikon API connector
- Estimated cost: £20–40K/yr per Bloomberg terminal seat

### IFRS 9 Hedge Accounting Documentation Generator

- Reads bias tracker + audit trail → generates structured hedge effectiveness documentation
- Required fields: hedge ratio, forecast bias within 80–125%, retrospective/prospective effectiveness test results
- Output: PDF / Word compatible with auditor review

### Mobile Dashboard (React Native / PWA)

- CFO needs mobile access for board meetings
- Minimum: Executive Summary + VaR gauge + top 3 InsightCards

### Slack / Teams Integration for Alerts

- Bias escalation (>10%) → Slack message to treasury team
- Change-point detection fire → Teams notification with commodity and direction
- Forecast refresh complete → summary pushed to configured channel

---

## 4. P2 — Wow / Enterprise Upsell (~12 months)

Capabilities that create new revenue streams and differentiate at the top end.

### Time-LLM / Chronos Zero-Shot Forecasting

- Integrate Chronos-2 (Oct 2025) or TimesFM as a zero-shot challenger
- Wrap in existing conformal layer for calibration guarantees
- Particularly valuable for 3 synthetic commodities (Rhodium, Polypropylene, ABS Resin) with limited exchange data

### Supply Chain Disruption Simulation (ML-based)

- Model cascading supplier failures using graph-based risk propagation
- Input: supplier exposure map + commodity correlation matrix
- Output: probability distribution of supply disruption → BOM cost impact

### Automated RFQ Price Negotiation Intelligence

- Given commodity forecast + current supplier quote, compute fair-value price range
- Feed into procurement negotiation playbook
- Potential partnership: Coupa, Jaggaer, Ivalua

### Carbon Cost Forecasting (Scope 3 Emissions)

- Extend BOM-weighted commodity index to include carbon intensity per tonne of material
- Integrate EU ETS carbon price forecast
- Output: forward-looking scope 3 emissions cost → EBIT impact under carbon pricing scenarios

### Weibull-Based Warranty Cost Insurance Premium Optimisation

- Extend current EWMA warranty model with Weibull hazard (shape k, scale λ)
- Compute fair-value warranty insurance premium as E[claim_cost] + risk-loading
- Enable CFO to decide: self-insure vs market insurance vs reinsurance blend

---

## 5. Immediate Quick Wins (1–2 days each)

Items that would visibly impress judges or CFO stakeholders with minimal effort.

| Item | Effort | Impact | How |
|---|---|---|---|
| **Live yfinance data replacing synthetic** | 1 day | Demo credibility | `scripts/fetch_data.py` already works; make it default and schedule daily |
| **Claude API for LLM narratives** | 1 day | Board-quality explanations | Replace template branch in `GICLLMEngine` with Anthropic SDK call |
| **PDF export of Executive Summary** | 2 days | CFO hand-off artifact | `reportlab` or `pdfkit`; render existing dashboard data as formatted PDF |
| **Email alert on bias escalation** | 1 day | IFRS 9 compliance demo | `smtplib` + existing `BiasTracker.compute_bias()` → email when `is_alert=True` |
| **Confidence interval overlay on all forecast charts** | 1 day | Visual credibility | Conformal module already produces intervals; pass to frontend `FanChart` component |

---

## 6. Technical Debt

Items that need cleanup before production — not features, but hygiene.

| Item | Severity | Description |
|---|---|---|
| **User store in-memory** | High | `auth/store.py` uses a dict; restarts lose sessions and users |
| **Audit trail flat file** | High | Single JSONL file doesn't scale; needs DB-backed storage for query + retention |
| **Test coverage gaps** | Medium | 34 tests pass but SOTA modules (conformal, change_point, quantile_forecast) lack unit tests for edge cases |
| **Config secret management** | High | `AUTH_SECRET_KEY` falls back to hardcoded dev secret; requires secrets manager (HashiCorp Vault / AWS Secrets Manager) in production |
| **Single-file CORS** | Medium | `allow_origins=["*"]` in dev config; must be scoped to customer domain in production |
| **No rate limiting** | Medium | FastAPI has no rate limiter; API endpoints are open to abuse without one |
| **WebSocket no auth** | Medium | `/ws/market` has no token check; any client can connect and receive live data |
| **Synthetic data default** | Medium | `data_source: synthetic` in config; production must flip to `parquet` or ERP connector and this should be caught at startup |
| **Model serialisation** | Low | XGBoost models saved with `joblib`; version pinning and schema versioning for model registry needed |
| **No pagination on audit trail** | Low | `get_recent_events(limit=100)` reads entire JSONL file; needs seek-from-end optimisation |
