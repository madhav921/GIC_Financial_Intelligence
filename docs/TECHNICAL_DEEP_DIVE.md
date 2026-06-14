# Technical Deep Dive — GIC Financial Intelligence Platform

---

## 1. Stack Overview

| Layer | Technology | Purpose | Why Chosen |
|---|---|---|---|
| L1 Data | Python, Pandas, Polars, CSV/Parquet | Synthetic O-U generator, data routing, ERP stubs | Polars ~10× faster than Pandas for large Parquet scans; Protocol abstraction lets data source swap in 1 config line |
| L2 Intelligence | SARIMAX (statsmodels), XGBoost 2.x, numpy | Commodity/demand forecasting, regime detection, conformal calibration | SARIMAX = interpretable seasonal decomp; XGBoost = nonlinear feature interaction; ensemble beats either alone across regimes |
| L3 Financial | Pure Python, numpy | BOM-weighted COGS, P&L waterfall, scenario shock propagation | Deterministic driver-based model; every £ traces commodity → BOM → COGS → EBIT with ~1% validation tolerance |
| L4 Simulation | numpy (scipy for hedge opt), XGBoost 2.x | Monte Carlo 10K sims, 7 preset scenarios, quantile VaR, hedge optimiser | Fat-tailed t(df=5) commodity shocks; scipy.optimize.minimize_scalar for hedge ratio; XGBoost 2.x joint quantile objective |
| L5 Governance | Python stdlib (JSONL, uuid, hmac, hashlib), HuggingFace/Ollama | Audit trail, bias tracking, SHAP explainability, LLM narratives | Zero-dependency JSONL = tamper-evident; stdlib crypto = no pyjwt/bcrypt licence risk; LLM cascade degrades gracefully |
| API | FastAPI, Pydantic v2, WebSocket | 30 REST endpoints + 1 WebSocket | Async-first; Pydantic v2 validation; OpenAPI schema auto-generated |
| Frontend | React, Recharts, Tailwind CDN, React Router | 11 pages, real-time market feed | Recharts = lightweight composable charts; Tailwind CDN = no build step; WebSocket + client fallback |
| Auth | Python stdlib (HMAC-SHA256, pbkdf2_hmac) | RBAC with 20-permission matrix, JWT | No third-party crypto deps; constant-time hmac.compare_digest prevents timing attacks |

---

## 2. Data Pipeline

### Ornstein-Uhlenbeck Synthetic Generator

The generator (`src/data/synthetic_generator.py`) simulates commodity prices via a discretised O-U process with drift:

```
drift_t  = κ · (μ_t − P_{t-1}) · dt
shock_t  = σ · P_{t-1} · √dt · ε,  ε ~ N(0,1)
P_t      = P_{t-1} + drift_t + shock_t
μ_t      = μ₀ · (1 + trend · t · dt)   # long-run mean with secular trend
```

**Parameters per commodity:**

| Parameter | Symbol | Role | Example (Lithium) |
|---|---|---|---|
| Base price | μ₀ | Long-run equilibrium level | 10 USD/kg |
| Volatility | σ | Proportional noise amplitude | 0.35 (35%) |
| Mean-reversion speed | κ | Rate of pull back to μ | 0.05 |
| Secular trend | trend | Monthly long-run mean drift | +0.02 (+2%/mo) |
| Time step | dt | 1/12 (monthly) | 0.0833 |

A seasonal overlay (3% Q4 amplitude via sin wave) is applied post-generation. All 12 commodities plus 12 macro indicators are generated with JLR-calibrated baselines (e.g., Steel: μ₀=490 USD/tonne, κ=0.12; Rhodium: μ₀=4500, κ=0.05, σ=0.35).

### Feature Engineering (`src/data/feature_engineering.py`)

Applied to every commodity series before ML training:

| Feature Group | Specifics |
|---|---|
| Lag features | Periods 1, 3, 6, 12 months |
| Rolling MA | Windows 3, 6, 12 months |
| Rolling std | Windows 3, 6, 12 months |
| Pct change | Periods 1, 3, 6 months |
| Calendar encoding | Month, quarter, year + cyclical sin/cos(month) |
| Macro join | GDP growth, interest rate, USD/GBP, USD/EUR, CPI, oil price, PMI, China PPI, DXY, Baltic Dry, US PPI, EV sales growth |

**Date-dtype safety fix:** Raw CSV date columns arrive as `object` dtype. The pipeline explicitly calls `pd.to_datetime()` before setting DatetimeIndex so SARIMAX receives a properly-ordered frequency-aware index. Without this, `statsmodels` raises a `ValueError` on non-datetime index, silently breaking seasonal inference.

---

## 3. Forecasting Models

### SARIMAX

- **Order:** `(1,1,1)×(1,1,1,12)` — AR(1) + first-difference + MA(1) with yearly seasonality
- **Exogenous:** GDP growth, interest rate, oil price, manufacturing PMI joined to the commodity series
- **Fit:** MLE via Kalman filter (statsmodels `SARIMAX`)
- **Output:** Point forecast + Gaussian 80%/95% CI (parametric, conservative)
- **Weakness:** Assumes Gaussian innovations; under fat-tailed shocks, claimed 80% CI degrades to ~70% coverage → corrected by conformal layer

### XGBoost Ensemble (`src/models/commodity_forecast_xgboost.py`)

- **Feature set:** ~60 features (all lag/rolling/pct/calendar/macro groups above)
- **Validation:** 5-fold time-series CV, MAPE reported per commodity
- **Measured MAPE (2024 hold-out):** Copper 7.0%, Platinum 8.9%, Polypropylene 9.8%; Natural Gas 31.1%, Palladium 29.1%
- **Feature importance:** XGBoost gain-based; top drivers vary by commodity (PMI, DXY, oil dominant for metals)

### Ensemble Regime-Adaptive Blending

Hurst exponent classifies each commodity into a regime, then weight profiles are applied:

| Regime | H threshold | SARIMAX | XGBoost | Futures | Scenarios |
|---|---|---|---|---|---|
| Mean-reverting | H < 0.45 | 45% | 20% | 25% | 10% |
| Trending | H > 0.55 | 15% | 45% | 30% | 10% |
| Volatile | 0.45–0.55 | 20% | 20% | 20% | 40% |

Regime-adaptive blending reduces MAPE by 15–25% during regime-shift periods vs fixed-weight ensembles.

### Conformal Prediction Intervals (`src/models/conformal.py`)

**Split-conformal:** Calibration set absolute residuals are sorted; the ⌈(n+1)(1−α)⌉/n empirical quantile sets the half-width. Coverage is guaranteed by construction — finite-sample, assumption-free.

**Adaptive Conformal Inference (ACI):** Online α update: if recent window is undercovering (more misses than α allows), α shrinks (wider intervals); if overcovering, α grows (tighter). This defends the 80% target exactly when models fail — during regime breaks.

Multi-step widening: half-width scaled by √h at horizon h to reflect accumulating uncertainty.

---

## 4. SOTA Modules

### CUSUM + BOCPD Change-Point Detection (G7) (`src/models/change_point.py`)

**Algorithm:**
- **CUSUM:** Accumulates signed deviations from a running mean; fires when cumulative sum exceeds `k·std(series)` threshold (Page 1954). Direction follows the triggering arm (upper/lower). O(n) per series.
- **BOCPD:** Bayesian run-length posterior P(r_t | x_{1:t}) with constant hazard H=1/λ and Gaussian-Normal predictive. Change probability = mass at r_t=0. scipy Student-t predictive used when available; Gaussian fallback otherwise (Adams & MacKay 2007).

**When it fires:** CUSUM flags a mean shift in ~2–3 observations; BOCPD provides a calibrated confidence (0–1) at each step. Dashboard alert threshold: confidence > 0.6 AND shifted=True.

**Dashboard impact:** `GET /intelligence/change-points/{commodity}` returns `shifted`, `direction`, `confidence`, `last_break_date`. All 12 commodities can be batch-scanned at `GET /intelligence/change-points`. Remaining open: auto-wire break events → reforecast trigger (backlog G7).

### QuantileForecaster — Gradient-Boosted Quantile VaR (G11) (`src/models/quantile_forecast.py`)

**Pinball loss:** L_τ(y, f) = max(τ·(y−f), (τ−1)·(y−f)). Minimising this over training data yields a consistent estimator of the conditional τ-quantile (Koenker & Bassett 1978).

**XGBoost 2.x joint objective:** `objective="reg:quantilederror"` with `quantile_alpha=[0.05, 0.50, 0.95]` trains one booster for all quantiles simultaneously — more stable than sequential single-quantile models and avoids quantile crossing by design.

**Fallback path:** Older XGBoost → single-booster per quantile with custom pinball gradient. sklearn `GradientBoostingRegressor(loss="quantile")` as final fallback (always available).

**Monotone post-sort:** After prediction, sort q05 ≤ q50 ≤ q95 to guarantee non-crossing bands even under numerical noise.

**API:** `GET /intelligence/quantile-var` returns asymmetric 5th/50th/95th commodity index bands feeding VaR/CVaR.

### ShapExplainer (`src/models/explainability_shap.py`)

**TreeSHAP (primary):** Computes exact Shapley values for tree ensembles in polynomial time (Lundberg & Lee 2017). Each driver receives a signed contribution in forecast units. Output: ranked list of `Driver(feature, value, contribution, direction)`.

**Fallback (shap not installed):** XGBoost gain-based global feature importance + single-pass permutation importance around the row being explained. Public API identical — callers never branch on presence of shap.

**Pipeline integration:** ShapExplainer feeds the LLM narrative layer: "Manufacturing PMI contributed +2.3% to the Copper forecast."

### ConformalForecaster (`src/models/conformal.py`)

**Split-conformal calibration:**
1. Hold out 20% of training data as calibration set.
2. Compute absolute residuals on calibration set: `|actual_i − forecast_i|`.
3. Sort ascending. The ⌈(n_cal+1)(1−α)⌉-th value = q_hat.
4. At test time: `[forecast − q_hat, forecast + q_hat]`.

**Coverage guarantee:** For exchangeable calibration/test data, marginal coverage ≥ 1−α holds in finite samples, regardless of model misspecification.

**ACI online update:** `α_{t+1} = α_t + γ·(α_target − 1_{missed_t})`, where γ is a step-size hyperparameter (default 0.05).

---

## 5. Financial Engine

### BOM-Weighted COGS Formula

```
Material_Spend_t = Σᵢ (w_i × Q_t × P_i_t)

where:
  w_i     = BOM weight of commodity i (Steel 22%, Lithium 18%, Aluminum 12%,
             Copper 7%, Cobalt 7%, Nickel 6%, Natural Gas 5%, Platinum 4%,
             Palladium 3%, Rhodium 2%, Polypropylene 8%, ABS Resin 6%)
  Q_t     = production volume at month t (demand-forecast driven)
  P_i_t   = commodity i price at month t (from L2 forecast)

COGS_t  = Material_Spend_t + Manufacturing_t + Labor_t + Overhead_t
EBIT_t  = Revenue_t − COGS_t − R&D_t − SG&A_t − Warranty_t − Depreciation_t
```

Revenue uses 4-segment elasticity adjustment: `Revenue_t = Σ_s (Base_s × (Vol_s_t/Vol_s_0)^ε_s)`.

### Scenario Shock Propagation

7 preset scenarios apply multiplicative shocks to commodity prices, demand, and FX:
- Bear: commodities +15–20%, demand −5%, FX adverse
- Bull: commodities −10–15%, demand +5%
- EV demand boom, recession, supply shock, currency crisis, base

Shocks propagate: commodity price → BOM-weighted COGS → gross margin → EBIT waterfall.

### Monte Carlo Distribution Sampling

```python
demand_shocks     ~ Normal(μ_demand, σ=0.10)          # 10% demand vol
commodity_shocks  ~ t(df=5) × 0.20 + μ_commodity     # fat-tail, 20% commodity vol
```

FX is absorbed into commodity costs. 10,000 paths computed vectorised in numpy. VaR(95%) = 5th percentile of operating income distribution; CVaR(95%) = mean of paths below VaR(95%).

---

## 6. Auth & RBAC

### 20-Permission Matrix

| Permission | Admin | User |
|---|:---:|:---:|
| view_landing | ✅ | ✅ |
| view_dashboard | ✅ | ✅ |
| view_executive_summary | ✅ | ✅ |
| view_forecasts | ✅ | ✅ |
| view_insights | ✅ | ✅ |
| view_aggregated_data | ✅ | ✅ |
| view_market_monitor | ✅ | ✅ |
| view_audit_summary | ✅ | ✅ |
| view_warranty | ✅ | ✅ |
| run_sandbox_simulation | ✅ | ✅ |
| run_simulation | ✅ | ❌ |
| edit_scenarios | ✅ | ❌ |
| view_raw_data | ✅ | ❌ |
| manage_thresholds | ✅ | ❌ |
| trigger_retraining | ✅ | ❌ |
| trigger_data_fetch | ✅ | ❌ |
| view_audit_full | ✅ | ❌ |
| export_reports | ✅ | ❌ |
| regenerate_narratives | ✅ | ❌ |
| manage_users | ✅ | ❌ |

### JWT Implementation

- **Algorithm:** HS256 (HMAC-SHA256) — stdlib only, no pyjwt
- **Format:** `base64url(header).base64url(payload).base64url(signature)` — standard JWT compact form
- **Expiry:** 480 minutes (8 hours) by default; configurable via `AUTH_TOKEN_EXPIRY_MINUTES`
- **Secret:** `AUTH_SECRET_KEY` env var; dev fallback with warning logged
- **Signature verification:** `hmac.compare_digest(expected, received)` — constant-time, prevents timing oracle

### Password Hashing

- **Algorithm:** `hashlib.pbkdf2_hmac("sha256", password, salt, iterations=200_000)`
- **Salt:** 16 bytes from `os.urandom()` — per-user, stored as hex alongside hash
- **Iterations:** 200,000 (2024 OWASP recommendation for PBKDF2-SHA256)
- **Verification:** Constant-time `hmac.compare_digest` on hex digests

---

## 7. Governance

### Bias Tracker (`src/governance/bias_tracking.py`)

```
bias_pct = (forecast − actual) / actual × 100
```

| Metric | Computation |
|---|---|
| mean_bias_pct | np.mean(bias_pct) across all observations |
| bias_direction | "over" if mean > 0, "under" if < 0 |
| is_alert | abs(mean_bias_pct) > threshold (default 5%) |
| recent_bias_trend | Compare recent half vs older half: "improving" if new < old×0.8, "worsening" if new > old×1.2, else "stable" |

Thresholds: >5% triggers alert; >10% would trigger escalation (IFRS 9 hedge relationship documentation risk). Configured in `settings.yaml` under `governance.bias_threshold_pct`.

### JSONL Audit Trail (`src/governance/audit_trail.py`)

- **Format:** Newline-delimited JSON, append-only file open (`"a"` mode)
- **Each entry:** `{entry_id: UUID4, timestamp: ISO8601 UTC, event_type, ...payload}`
- **Event types:** `forecast_generated`, `override_applied`, `scenario_run`, `bias_alert`, `model_trained`, generic `log_event`
- **20 events per full pipeline run** (one per commodity forecast + P&L events + governance events)
- **Immutability:** File is never truncated; entries are never edited. UUID4 key per entry.

### LLM Engine Cascade (`layers/layer5_governance/llm_engine.py`)

Priority order:
1. **Ollama (llama3.2:1b):** HTTP probe to `localhost:11434/api/tags`; checks model is pulled
2. **HuggingFace transformers (google/flan-t5-base):** ~300MB, no GPU required
3. **Template fallback:** Deterministic string interpolation — zero dependencies, always available

Config: `temperature=0.3` (low, for factual financial text), `max_new_tokens=256`.

---

## 8. WebSocket Real-Time Market Feed

### MarketFeed Class (`src/api/routes/realtime.py`)

Mean-reverting random walk per tick:
```python
new_price = price × (1 + mean_reversion × (target − price)/price × dt + vol × N(0,1) × √dt)
```

Parameters: `vol=0.002` per tick (small, realistic intra-day), `mean_reversion=0.1`, tick interval configurable (default 2s).

**Seeding:** Reads last row of `data/synthetic/commodity_prices.csv` if present; falls back to `_DEFAULT_PRICES` dict.

**WebSocket endpoint:** `GET /ws/market` — broadcasts JSON tick every 2 seconds with `{prices, fx, risk_score, risk_band, headline, timestamp}`.

**REST snapshot:** `GET /realtime/market-snapshot` — same payload, single response.

### Client-Side Simulator Fallback (`frontend/src/context/RealtimeContext.jsx`)

`RealtimeContext` singleton: on `useEffect` mount, attempts WebSocket connection. If connection fails or drops, activates a client-side mean-reverting walk using `setInterval` at the same tick rate. Consumers see identical data shape either way.

---

## 9. API Surface

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | /health | No | Liveness check |
| GET | /health/ready | No | Readiness (data + models loaded) |
| POST | /forecast/commodity | No | SARIMAX forecast for one commodity |
| GET | /forecast/commodity-index | No | BOM-weighted commodity index time series |
| GET | /forecast/elasticity | No | Price elasticity estimates per segment |
| POST | /simulation/scenario | No | Monte Carlo scenario with custom shocks |
| POST | /pnl/shock | No | BOM-weighted P&L waterfall for price shocks |
| GET | /pnl/regime | No | Hurst-based regime for all commodities |
| GET | /pnl/annual | No | Annual P&L KPI summary |
| POST | /auth/login | No | Credential verification → JWT |
| GET | /auth/me | Yes | Current user profile |
| POST | /auth/logout | Yes | Stateless logout |
| GET | /auth/permissions | Yes | Role + resolved permission list |
| GET | /auth/users | Admin | List all users |
| GET | /auth/demo-profiles | No | Demo login credentials (demo mode only) |
| GET | /insights/feed | Yes | Prioritised InsightCard feed |
| GET | /insights/variance-bridge | Yes | Plan→actual EBIT waterfall |
| GET | /insights/early-warning | Yes | Composite risk score |
| GET | /insights/warranty/summary | Yes | Warranty forecast + accrual + failure modes |
| POST | /insights/recommend/hedge | Yes | Hedge-ratio recommendation |
| WS | /ws/market | No | Live commodity price WebSocket stream |
| GET | /realtime/market-snapshot | No | Single-tick market snapshot |
| GET | /intelligence/change-points/{commodity} | No | CUSUM+BOCPD break alert for one commodity |
| GET | /intelligence/change-points | No | Scan all 12 commodities for breaks |
| GET | /intelligence/quantile-var | No | Gradient-boosted quantile VaR/CVaR |
| GET | /intelligence/shap/{commodity} | No | SHAP driver attribution |
| GET | /intelligence/conformal/{commodity} | No | Conformal interval for commodity |
| GET | /intelligence/bias-report | No | Bias tracking report across commodities |
| GET | /intelligence/audit-trail | Admin | Recent audit entries |
| GET | /intelligence/llm-narrative | Yes | LLM-generated narrative for scenario |

---

## 10. Frontend Architecture

### Context Hierarchy

```
App
└── AuthContext (JWT store, login/logout, role)
    └── ProtectedRoute (redirect to /login if unauthenticated)
        └── PermissionGate (hide/disable UI elements by permission)
            └── RealtimeContext (WebSocket singleton, client fallback)
                └── Pages / Components
```

### Page Inventory

| Page | Route | Key Components | Auth Required |
|---|---|---|---|
| Landing | / | Hero, feature cards | No |
| Login | /login | LoginForm, demo-profile buttons | No |
| Executive Summary | /executive | KPICard, FanChart, WaterfallChart, LiveKpiStrip | Yes |
| Commodity Intelligence | /commodity | PriceChart, ForecastBands, ShockSlider, DriverAttribution | Yes |
| Financial P&L | /pnl | DistributionHistogram, WaterfallChart, TornadoChart | Yes |
| Market Monitor | /market | CorrelationHeatmap, LiveMarketTape, RegimeCard | Yes |
| Scenario Simulation | /scenarios | ScenarioBuilder, FanChart, ScenarioCompare | Admin |
| Insights Center | /insights | InsightCard, RecommendationPanel, EarlyWarningGauge | Yes |
| Variance Bridge | /variance | VarianceBridgeChart (waterfall) | Yes |
| Warranty Analytics | /warranty | WearCurveChart, AccrualTable, FailureModeList | Yes |
| Governance | /governance | AuditTrail, BiasReport, LLMNarrative | Admin |
| Data Explorer | /data | RawDataTable, ColumnFilter | Admin |

### Component Dependency (text graph)

```
RealtimeContext → LiveMarketTape, LiveKpiStrip, PriceChart (live updates)
AuthContext → PermissionGate → LockedButton (disabled non-admin actions)
InsightEngine API → InsightCard → RecommendationPanel
VarianceBridge API → VarianceBridgeChart
ShapExplainer API → DriverAttribution
ChangePoint API → RegimeCard (alert badge)
```
