# Why / How / Impact — GIC Feature Reference

*For every major implemented feature: What it is | Why we built it | How it works | Measurable impact.*

---

### 1. SARIMAX + XGBoost Ensemble Forecasting

**What:** Regime-adaptive blend of a seasonal ARIMA model with macroeconomic exogenous inputs (SARIMAX) and a gradient-boosted tree model (XGBoost), weighted dynamically by the current market regime.

**Why:** No single model dominates across all commodity regimes. SARIMAX captures mean-reversion and seasonality; XGBoost captures nonlinear macro-driven trends. Fixed-weight blends underperform during regime transitions.

**How:** SARIMAX fitted with order (1,1,1)×(1,1,1,12) on monthly price history with macro exogenous (GDP, PMI, oil, DXY). XGBoost trained on 60+ engineered features (lags 1/3/6/12, rolling MA/std 3/6/12, pct change, calendar, all macro). Hurst exponent classifies regime → weights applied (e.g., trending: SARIMAX 15%, XGBoost 45%, futures 30%, scenarios 10%). Output: 12-month point forecast.

**Impact:** Best MAPE 7.0% (Copper), 8.9% (Platinum); regime-adaptive blending reduces MAPE by 15–25% vs fixed weights during regime-shift periods. 5-fold CV prevents overfitting.

---

### 2. Adaptive Conformal Prediction Intervals

**What:** Distribution-free prediction intervals that are provably calibrated to the stated coverage level, wrapping any point forecaster. Two modes: split-conformal (static) and Adaptive Conformal Inference / ACI (online-updating).

**Why:** SARIMAX parametric CIs assume Gaussian innovations; under fat-tailed commodity shocks they achieve only ~70% coverage when 80% is stated. Conformal intervals are calibrated empirically — they deliver the promised coverage by construction.

**How:** Split-conformal: absolute residuals from a held-out calibration set are sorted; the ⌈(n+1)(1−α)⌉-th value sets the half-width. ACI: each step, if the prediction missed, α decreases (wider); if it covered, α increases (tighter). Online update: α_{t+1} = α_t + γ(α_target − 1{missed_t}). Multi-step: half-width scaled by √h at horizon h.

**Impact:** Coverage guarantee holds in finite samples, regardless of model misspecification — the strongest theoretical guarantee available for prediction intervals. Particularly valuable during regime breaks when commodity forecasters systematically fail. Module: `src/models/conformal.py`.

---

### 3. CUSUM + BOCPD Change-Point Detection (G7)

**What:** Two complementary online structural-break detectors that fire when commodity price series undergo abrupt regime shifts — earlier than the Hurst exponent which needs a full window (~12 obs) in the new regime.

**Why:** A 2-week-late regime detection means 2 weeks of stale forecasts and unhedged exposure. Early warning enables faster reforecast and rehedge decisions.

**How:** CUSUM (Page 1954): accumulates signed deviations from running mean; fires when cumulative sum exceeds k·std(series). Direction (up/down) follows the triggering arm. O(n). BOCPD (Adams & MacKay 2007): maintains Bayesian run-length posterior P(r_t | x_{1:t}) with constant hazard H=1/λ and Gaussian-Normal predictive; change probability = mass at r_t=0. scipy Student-t predictive used when available; Gaussian fallback otherwise.

**Impact:** Dashboard alert in 2–3 observations after a mean shift; BOCPD provides calibrated confidence (0–1) enabling threshold-based escalation. API: `GET /intelligence/change-points/{commodity}`. Module: `src/models/change_point.py`.

---

### 4. Gradient-Boosted Quantile VaR (G11)

**What:** XGBoost 2.x trained to directly minimise pinball (quantile) loss at τ=0.05, 0.50, 0.95 — producing asymmetric prediction bands that honestly represent skewed commodity tail risk.

**Why:** Symmetric Gaussian VaR assumes equal upside/downside — incorrect for metals (Lithium spikes violently on supply shocks, grinds down slowly). Asymmetric quantiles correctly price fat downside tails.

**How:** XGBoost 2.x `objective="reg:quantilederror"`, `quantile_alpha=[0.05, 0.50, 0.95]` trains one booster for all quantiles jointly (avoiding quantile crossing). Older XGBoost fallback: one booster per quantile with custom pinball gradient. sklearn `GradientBoostingRegressor(loss="quantile")` as always-available fallback. Monotone post-sort guarantees q05 ≤ q50 ≤ q95.

**Impact:** Asymmetric 5th/95th bands feed VaR/CVaR computation; wider on the downside for supply-shock-prone commodities. API: `GET /intelligence/quantile-var`. Module: `src/models/quantile_forecast.py`.

---

### 5. SHAP Feature Attribution

**What:** Game-theoretic Shapley values computed per-forecast, attributing each driver's signed contribution to the XGBoost commodity prediction. Output: ranked list of drivers ("Manufacturing PMI contributed +2.3% to Copper forecast").

**Why:** Opaque ML models lose board trust. CFOs and auditors need to understand why the model said what it said — not just what it said.

**How:** `shap.TreeExplainer` computes exact Shapley values for tree ensembles in polynomial time (Lundberg & Lee 2017). Each feature receives a signed contribution in forecast units. Fallback (shap not installed): XGBoost gain-based global importance + single-pass permutation importance around the row being explained. Public API identical — callers never branch on shap presence.

**Impact:** Turns every commodity forecast into a cited explanation that feeds the LLM narrative layer and the SHAP driver widget in the UI. Reduces model-trust barrier for CFO sign-off. Module: `src/models/explainability_shap.py`.

---

### 6. Monte Carlo Simulation (Fat-Tail t-Distribution)

**What:** 10,000-path stochastic simulation drawing demand from Normal and commodity shocks from t(df=5) distribution to generate a full probability distribution of EBIT outcomes.

**Why:** Normal distribution assumption fails during commodity shocks (e.g., Lithium crash, NatGas spike). You miss tail risk — the most expensive thing to miss. t(df=5) has fatter tails than Normal, capturing extreme events more honestly.

**How:** `demand_shocks ~ N(μ, 0.10)`. `commodity_shocks ~ t(df=5) × 0.20 + μ_commodity`. Each path: revenue = base_revenue × (1 + demand_shock); COGS = base_cogs × (1 + demand_shock) + base_cogs × material_fraction × commodity_shock; EBIT = margin − fixed_costs. All 10,000 paths computed vectorised in numpy (~0.2s). VaR(95%) = 5th percentile of EBIT distribution; CVaR(95%) = mean of paths below VaR.

**Impact:** VaR95 = £19.3bn, CVaR95 = £8.3bn on £176bn revenue base. Risk decomposition: commodity 66%, FX 26%, demand 8%. Calibrated to 79% empirical coverage vs 80% target (PASS). Module: `src/simulation/monte_carlo.py`.

---

### 7. Hurst Exponent Regime Detection

**What:** Rescaled-range (R/S) statistic classifying each commodity series into MEAN_REVERTING (H<0.45), TRENDING (H>0.55), or VOLATILE (0.45–0.55) to drive ensemble weight switching.

**Why:** A single model family never wins across all regimes. Lithium in 2022 was trending (XGBoost dominant); in 2024 mean-reverting (SARIMAX dominant). Misclassification wastes model capacity and inflates MAPE.

**How:** Hurst H ∈ [0,1] computed via R/S analysis on the commodity price series. H≈0.5 = random walk; H<0.45 = anti-persistent (mean-reverting); H>0.55 = persistent (momentum). Regime determines ensemble weight profile applied to SARIMAX + XGBoost + futures + scenarios blend.

**Impact:** 15–25% MAPE reduction during regime-shift periods vs fixed-weight ensembles (documented in module rationale). Enables the platform to self-adapt without manual model selection. Module: `src/models/regime_detector.py`.

---

### 8. BOM-Weighted Commodity Index

**What:** A single time-series index representing the weighted-average price movement of all 12 commodities in the vehicle Bill of Materials, expressed as % change vs a base period.

**Why:** OEM commodity exposure is a portfolio, not individual positions. The relevant risk metric is the weighted portfolio move, not any individual commodity. The index is the fundamental input to COGS calculation.

**How:** `Index_t = Σᵢ (w_i × P_i_t / P_i_base)` where w_i are BOM weights summing to 1.0 (Steel 22%, Lithium 18%, Aluminum 12%, etc.). Index feeds deterministic COGS: `Material_Spend_t = Σᵢ (w_i × Q_t × P_i_t)`. Weights configured in `config/settings.yaml`.

**Impact:** Single-number commodity risk metric for the CFO dashboard; every 1pp index move has a quantified EBIT impact. Currently tracking +84.7% YoY on latest data run. Module: `src/models/commodity_forecast.py::generate_commodity_index`.

---

### 9. Ornstein-Uhlenbeck Synthetic Data Generator

**What:** Discretised O-U process with secular drift generating 84 months of realistic synthetic commodity price series for 12 commodities and 12 macro indicators.

**Why:** Real ERP data requires 3–6 month vendor integration. A calibrated synthetic generator enables immediate demos, testing, and development without exposing real customer data.

**How:** `P_t = P_{t-1} + κ(μ_t − P_{t-1})dt + σ·P_{t-1}·√dt·ε` where ε ~ N(0,1). Parameters per commodity: base price μ₀, volatility σ, mean-reversion speed κ, secular trend. Seasonal overlay (3% amplitude, sin(2π·t/12)) applied post-generation. Seed=42 for reproducibility. 12 commodity + 12 macro series generated simultaneously with correlation structure implicit in shared macro context.

**Impact:** Enables full platform demo with realistic price dynamics (trend, mean-reversion, seasonality, fat tails) without any external data dependency. Module: `src/data/synthetic_generator.py`.

---

### 10. RBAC with 20-Permission Matrix

**What:** Role-based access control system with 2 roles (Admin, User) and 20 granular permissions controlling which API endpoints and UI elements each user can access.

**Why:** Enterprise deployments require that analysts cannot trigger model retraining or access raw data, while admins have full control. Without RBAC, a single compromised analyst credential exposes the entire platform.

**How:** `Permission` enum (20 values) + `Role` enum (Admin/User) in `auth/models.py`. `ROLE_PERMISSIONS` dict maps each role to a set of permissions. `has_permission(role, permission)` is the single check used everywhere. FastAPI `Depends(get_current_user)` + `Depends(require_admin)` enforce at the route level. Frontend `PermissionGate` component hides/disables UI elements for insufficient permissions; `LockedButton` shows disabled state with tooltip.

**Impact:** Clean separation between CFO view (read-only, User role) and data/model management (Admin). 10 permissions gated to Admin prevent misuse of simulation and raw-data endpoints. Module: `auth/permissions.py`, `auth/dependencies.py`.

---

### 11. Immutable JSONL Audit Trail

**What:** Append-only newline-delimited JSON log of every forecast generation, scenario run, override, and governance event — UUID-keyed, UTC-timestamped, never modified.

**Why:** IFRS 9 hedge accounting requires documented evidence that forecast assumptions and hedge ratios were determined based on consistent, recorded methodology. Without an audit trail, model override events are invisible to auditors.

**How:** File opened with mode `"a"` (append-only). Each entry: `{"entry_id": UUID4, "timestamp": ISO8601_UTC, "event_type": ..., ...payload}`. Entry types: `forecast_generated`, `override_applied`, `scenario_run`, `bias_alert`, `model_trained`, `log_event`. `json.dumps(entry, default=str)` handles datetime serialisation. 20 events per full pipeline run.

**Impact:** Board-ready audit trail for IFRS 9 documentation. UUID-keyed entries can be referenced by auditors. Immutability (append-only) provides tamper-evidence. Module: `src/governance/audit_trail.py`.

---

### 12. Bias Tracking with Governance Escalation

**What:** Systematic tracking of forecast bias (mean % over/under-forecasting) per commodity per model, with alert thresholds and trend direction classification.

**Why:** Systematic bias in commodity forecasts directly threatens IFRS 9 hedge effectiveness. A model that consistently under-forecasts Lithium prices by 12% is providing misleading input to hedge-ratio decisions — and may disqualify hedge accounting.

**How:** `bias_pct = (forecast − actual) / actual × 100`. Mean bias, median bias, direction ("over"/"under"), `is_alert` if `abs(mean_bias) > threshold` (default 5%). Trend: compare recent half vs older half — "improving" if new abs bias < old × 0.8, "worsening" if > 1.2×. Escalation threshold at 10% (configurable) signals CFO-level review requirement.

**Impact:** Automated governance check replacing manual quarterly review. >5% bias → dashboard alert; >10% → escalation workflow. Bias report feeds IFRS 9 documentation package. Module: `src/governance/bias_tracking.py`.

---

### 13. Open-Source LLM Cascade (Ollama → HuggingFace → Template)

**What:** Three-tier LLM backend that auto-detects the best available language model and gracefully degrades — from local Ollama (llama3.2:1b) to HuggingFace transformers (google/flan-t5-base) to deterministic template — with zero code changes for callers.

**Why:** Enterprise deployments vary: some have GPU servers (Ollama), some have CPU-only cloud (HuggingFace), some have no LLM infrastructure (template). A cascade ensures narrative generation always works.

**How:** `_init_backend()`: HTTP probe to `localhost:11434/api/tags` → if model present, use Ollama. Else: attempt `from transformers import pipeline` → if available, load flan-t5-base. Else: template strings with f-string interpolation. `temperature=0.3` for factual financial text; `max_new_tokens=256`. All backends expose identical `explain_forecast()`, `generate_risk_summary()`, `explain_alert()` methods.

**Impact:** Governance narrative generation works in any deployment environment. Ollama path produces quality prose; template path produces structured, factual output. Upgrade path: replace template branch with Claude API call (1-day effort, see ROADMAP.md). Module: `layers/layer5_governance/llm_engine.py`.

---

### 14. Plan-to-Perform Variance Bridge (EBIT Waterfall)

**What:** Structured decomposition of the gap between planned EBIT and actual/forecast EBIT into causal drivers: Volume, Price/Mix, Commodity, FX, Warranty, and Other.

**Why:** A single EBIT miss number ("EBIT was £99M below plan") tells the CFO nothing actionable. The bridge identifies whether the miss is demand-driven (Volume), pricing-driven, or commodity-driven — each requiring a different management response.

**How:** Sequential waterfall: Volume bridge = (actual_volume − plan_volume) × plan_margin_per_unit. Price/Mix = (actual_price − plan_price) × actual_volume. Commodity = −(actual_commodity_cost − plan_commodity_cost). FX = FX-adjusted revenue delta. Warranty = actual_warranty_provision − plan. Other = residual. Implemented via `VarianceBridgeAnalyzer.build_from_scenarios()`. Rendered as waterfall chart in frontend `VarianceBridgeChart`.

**Impact:** CFO-standard format (standard in automotive management reporting). Enables board conversation about which levers to pull. Example from code: plan £1.5bn vs actual £1.401bn (−£99M) decomposed into 6 drivers. Module: `src/insights/variance_bridge.py`.

---

### 15. Warranty Analytics (EV Learning Curve + Weibull)

**What:** Forward-looking warranty cost model combining an EV battery learning curve (cost declining with cumulative volume) with time-series warranty incident tracking and accrual adequacy assessment.

**Why:** EV warranty costs are structurally different from ICE — battery pack failures have different failure modes, longer warranty tails, and a cost-improvement learning curve that affects future accrual adequacy. Under-accruing warranty is a material misstatement risk.

**How:** Historical warranty_data.csv generated by `WarrantyDataGenerator` with: incident_date, vehicle_segment, cost_per_incident, repair_type. EV learning curve: `cost_t = cost_0 × (cumulative_volume_t / V_0)^{−b}` where b ≈ 0.15 (Wright's Law). Warranty summary endpoint aggregates: total_cost, incidents_by_segment, cost_per_vehicle, learning_curve_projection, accrual_adequacy (actual vs provision).

**Impact:** Early detection of accrual shortfalls; EV learning curve projection shows when battery costs will normalise. API: `GET /insights/warranty/summary`. Module: `src/models/warranty_model.py`, `src/data/warranty_generator.py`.

---

### 16. WebSocket Real-Time Market Feed

**What:** FastAPI WebSocket endpoint broadcasting commodity prices, FX rates, a composite risk score, and curated headlines every 2 seconds — with a client-side mean-reverting fallback if the WebSocket drops.

**Why:** CFOs and traders need to see live commodity moves in context of their EBIT exposure. A 2-second live tape creates the "Bloomberg terminal feel" that makes the platform credible in executive demos.

**How:** Server: `GET /ws/market` streams JSON ticks. Each tick: seed from last CSV row → apply mean-reverting walk `(1 + κ(μ−P)/P·dt + σ·N(0,1)·√dt)` with vol=0.002 (small, realistic). Risk score: composite of price deviation from 30-day MA across 6 commodities → mapped to `low/elevated/high/critical` band. Client: `RealtimeContext.jsx` attempts WebSocket on mount; if it fails, runs identical math client-side via `setInterval`. Consumers never know which path is active.

**Impact:** Demo-ready real-time feed with no external market data dependency. Realistic price dynamics (bounded, mean-reverting). Module: `src/api/routes/realtime.py`, `frontend/src/context/RealtimeContext.jsx`.

---

### 17. InsightCard + Recommendation Engine

**What:** Prioritised feed of prescriptive, £-quantified action cards generated from the commodity forecast, Monte Carlo output, and warranty analytics — ranked by financial impact.

**Why:** Analytics without prescription is half a product. A CFO seeing a risk should immediately know what to do about it and what it costs to act vs not act.

**How:** `InsightEngine` reads commodity forecast outputs, variance bridge, early warning score, and warranty summary. For each relevant signal, generates an `InsightCard(title, description, impact_gbp, action, priority, category)`. Priority: HIGH for signals >£50M impact or bias alerts. Cards sorted by |impact_gbp| descending. `summary_stats()` aggregates total_impact, high_priority_count. API: `GET /insights/feed?top_n=8`.

**Impact:** CFO gets 8 ranked actions with £ impact in a single API call. Prescriptive vs analytical — closes the "so what?" loop. Module: `src/insights/insight_engine.py`, `src/insights/recommendation_engine.py`.

---

### 18. Hedge Optimiser (Portfolio-Theory Optimal Ratio)

**What:** Computes the optimal hedge ratio h* ∈ [0,1] for a commodity exposure that minimises a blend of expected procurement cost and VaR(95%), given ML forecast mean/std, current futures price, and hedge cost.

**Why:** Static "hedge 50%" rules ignore forecast uncertainty, futures basis, and hedge costs. Portfolio-theory optimisation balances expected savings against tail protection — the same approach used by commodity trading desks.

**How:** `cost(h) = (1−h)·E[P]·units + h·futures_price·(1 + cost_bps)·units`. `VaR(h) = (1−h)·(E[P] + z_{0.95}·σ)·units + h·futures_price·(1+cost_bps)·units`. Objective: `min_h [α·cost(h) + (1−α)·VaR(h)]`, α=0.5 default. Solved via `scipy.optimize.minimize_scalar` on [0,1]. Output: `{hedge_ratio, expected_savings, var_reduction, recommendation}`.

**Impact:** £1.5M/yr expected savings vs industry-standard static 50% ratio (from ARCHITECTURE_GUIDE.md; Aluminum at optimal 75% vs 50% static). Quantified recommendation the treasury team can act on immediately. Module: `src/models/hedge_optimizer.py`.
