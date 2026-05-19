# Architecture Deep Dive — GIC Platform

This guide explains the 5-layer architecture, the role of each component, and how data flows end-to-end.

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: DATA INGESTION & CACHING                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input Sources          Pipeline                  Storage                        │
│  ──────────────         ────────                  ───────                        │
│                                                                                  │
│  • Yahoo Finance   ┐                                                             │
│  • FRED (macro)   │    ┌──────────────────┐      ┌──────────────────┐           │
│  • CCXT (crypto)  ├───▶│ Polars Pipeline │─────▶│ Parquet Files    │           │
│  • SAP/SF (stubs) │    │ (src/data/)      │      │ (data/external/) │           │
│  • Synthetic gen  │    └──────────────────┘      └──────────────────┘           │
│                   ┘     Config-driven              ↓                             │
│                         data routing          DataFrame cache                    │
│                                                                                  │
│  Result: Single abstraction layer (Protocol interface) — swap data sources in    │
│          1 config line. All downstream code is source-agnostic.                  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              LAYER 2: PREDICTIVE INTELLIGENCE (ML Forecasting)                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Commodity Forecast (12 materials)                                              │
│  ───────────────────────────────────                                            │
│                                                                                  │
│  For each commodity:                                                             │
│                                                                                  │
│    Historical prices  ┐                                                          │
│    + Macro context    │                                                          │
│    + Regime signal    ├─ Regime Detection (Hurst exponent)                      │
│                       │  └─ Classify: Trending vs. Mean-reverting                │
│                       │                                                          │
│    ┌─ Adaptive Weighting ┐                                                      │
│    │  • If Trending       │ ▶ XGBoost MAPE: 15.3%  (reg. weight: 60%)           │
│    │  • If Mean-reverting │                                                      │
│    │  • Recent accuracy   │ ▶ SARIMAX MAPE: 18.7%  (reg. weight: 25%)           │
│    │    track            │                                                       │
│    │  • Futures signal    │ ▶ Futures curve (reg. weight: 15%)                  │
│    └─ Result: Adaptive ensemble weights                                         │
│                                                                                  │
│  Output: 12-month forecast + 95% confidence bands (SARIMAX volatility)          │
│          + Directional accuracy (% chance of up/down being correct)              │
│                                                                                  │
│  Demand Forecast (4 segments)                                                    │
│  ─────────────────────────────                                                  │
│                                                                                  │
│  • XGBoost on 60+ features (macro, pricing, calendar, elasticity)               │
│  • Per-segment (EV, Luxury SUV, Performance, Premium SUV)                       │
│  • 12-month forecast + bands                                                    │
│                                                                                  │
│  Price Elasticity Model                                                          │
│  ───────────────────────                                                         │
│                                                                                  │
│  • Log-log regression: ln(Demand) = α + β×ln(Price) + γ×ln(Competitor_Price) + ...
│  • Captures own-price elasticity + cross-commodity elasticity                    │
│  • Used for scenario sensitivity                                                │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│              LAYER 3: FINANCIAL DRIVERS (Deterministic P&L Model)                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input: Commodity forecasts + demand forecasts                                  │
│  ────────────────────────────────────────────                                   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │ REVENUE SIDE                                                        │         │
│  ├─────────────────────────────────────────────────────────────────────┤         │
│  │                                                                     │         │
│  │  Base Revenue = £24B                                               │         │
│  │  Elasticity-adjusted Revenue = Base × (Forecast_Demand_t /        │         │
│  │                                        Base_Demand_0)^elasticity   │         │
│  │                                                                     │         │
│  │  Segments: EV (25%), Luxury SUV (35%), Performance (20%), Prem (20%)        │
│  │                                                                     │         │
│  └─────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │ COGS SIDE                                                           │         │
│  ├─────────────────────────────────────────────────────────────────────┤         │
│  │                                                                     │         │
│  │  Material Spend = Σ (BOM_Weight_i × Quantity_Produced × Price_i)  │         │
│  │                                                                     │         │
│  │  BOM_Weights: Steel 22%, Lithium 18%, Aluminum 12%, ... (12 total)│         │
│  │  Quantity: Driven by demand forecast (volume elasticity)          │         │
│  │  Prices: From Layer 2 commodity forecasts                         │         │
│  │                                                                     │         │
│  │  COGS = Material_Spend + Manufacturing + Labor + Overhead         │         │
│  │                                                                     │         │
│  └─────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │ PROFITABILITY DRIVERS                                               │         │
│  ├─────────────────────────────────────────────────────────────────────┤         │
│  │                                                                     │         │
│  │  Gross Margin % = (Revenue - COGS) / Revenue                      │         │
│  │                                                                     │         │
│  │  Fixed Costs: Warranty (1.8% of revenue), Deprec. (2.2%)          │         │
│  │                                                                     │         │
│  │  EBIT = Gross Margin - Fixed Costs - R&D - SG&A                  │         │
│  │                                                                     │         │
│  │  EBIT Margin = EBIT / Revenue  (target: 18%+)                     │         │
│  │                                                                     │         │
│  └─────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  Output: Base case P&L, Bear/Base/Bull scenarios, sensitivity tables             │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│            LAYER 4: RISK & SIMULATION (Monte Carlo, Hedge Optimization)           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Monte Carlo Engine (10,000 simulations)                                        │
│  ───────────────────────────────────────                                        │
│                                                                                  │
│  For each simulation path:                                                      │
│                                                                                  │
│    1. Sample commodity returns from multivariate normal (with observed          │
│       correlations + fat-tail mixture component)                                │
│    2. Simulate demand shocks (EV adoption, macro slowdown, etc.)               │
│    3. Propagate through financial model → P&L distribution                      │
│    4. Compute VaR (95%), CVaR (95%), and margin distribution                   │
│                                                                                  │
│  Result: Probabilistic P&L instead of point forecast                            │
│          EBIT range: £1.2B – £1.6B (80% CI)                                     │
│          VaR(95%): £705M downside                                               │
│          Calibrated to 79% actual coverage (robust)                              │
│                                                                                  │
│  Hedge Optimization (using portfolio theory)                                    │
│  ──────────────────────────────────────────                                     │
│                                                                                  │
│  Problem: Which commodities should we hedge, and how much?                      │
│                                                                                  │
│  Solution: Minimize variance of EBIT subject to:                                │
│    • Hedge cost constraints (can't hedge 100% of everything)                    │
│    • Liquidity constraints (limited futures volume per commodity)               │
│    • Portfolio approach: co-variances matter (avoid over-hedging correlated)     │
│                                                                                  │
│  Output: Optimal hedge ratio per commodity (e.g., Aluminum: 75% vs. 50% static) │
│          Expected savings: £1.5M/yr vs. industry standard                       │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│           LAYER 5: GOVERNANCE & EXPLAINABILITY (Audit Trail, Transparency)        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Audit Trail (JSONL logs)                                                       │
│  ─────────────────────────                                                      │
│  • Every forecast, every P&L calc, every override logged with timestamp         │
│  • Who changed what, when, and why (immutable trail)                            │
│  • Compliance ready for CFO reviews                                             │
│                                                                                  │
│  Explainability Engine                                                           │
│  ────────────────────                                                           │
│  • Feature importance: Which economic factors drove the forecast?               │
│  • Scenario decomposition: "Lithium +20% = EBIT -£180M. Why?"                  │
│  • Model selection: "We chose XGBoost (60%) over SARIMAX (40%) because..."      │
│  • Bias tracking: Does the model underestimate risks in certain regimes?        │
│                                                                                  │
│  Report Generation                                                               │
│  ────────────────                                                               │
│  • Executive Intelligence Report (766 lines, board-ready)                       │
│  • Contains: market snapshot, forecasts, P&L impacts, risks, recommendations    │
│  • Regenerated daily or on-demand with latest data                              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions & Why

### 1. **Protocol-Based Data Abstraction**
**Problem**: Forecasting models are useless without data. Production financial systems use SAP/Salesforce, but these require long integration timelines and restricted access. Meanwhile, real Yahoo Finance data is available immediately and serves as a validated proxy.

**Solution**: Define abstract interfaces (`MarketDataSource`, `OperationalDataSource`) that any implementation can conform to. Swap between real Yahoo Finance (current default), synthetic JLR-calibrated data (for offline demos), and production SAP (future — one config line change).

**Why "JLR-calibrated"?** Synthetic operational data (BOM weights, vehicle segments, production volumes, warranty rates) is calibrated to match the scale and structure of a Jaguar Land Rover-sized automotive company: ~375,000 annual vehicle production, £24B revenue, 4 segments (EV, Luxury SUV, Performance, Premium SUV). This makes the financial model realistic even without real ERP data.

**File**: `src/data/data_source_protocol.py`

---

### 2. **Adaptive Ensemble Forecasting**
**Problem**: Single-model forecasts (ARIMA or ML) fail during market regime shifts. Lithium in 2023 was mean-reverting; 2024 is trending. One-size-fits-all doesn't work.

**Solution**: Detect the regime (Hurst exponent), then adaptively reweight 4 models:
- XGBoost (if trending)
- SARIMAX (if mean-reverting)
- Futures curve (if liquid)
- Sentiment (social media signal)

**Result**: 22% MAPE improvement in 2024 vs. naive single-model approach.

**File**: `src/models/regime_detector.py`, `src/models/commodity_forecast.py`

---

### 3. **Monte Carlo with Fat Tails**
**Problem**: Normal distribution assumption fails during commodity shocks (e.g., Lithium crash). You miss tail risk.

**Solution**: Use a mixture of normal + Student-t distribution in Monte Carlo. Captures both baseline volatility AND extreme events.

**Result**: VaR estimates stay consistent even during market dislocations.

**File**: `src/simulation/monte_carlo.py`

---

### 4. **Scenario-Based Planning**
**Problem**: Point forecasts (EBIT = £1.4B) are useless for CFO planning. They're always wrong.

**Solution**: 7 pre-baked scenarios (commodity shocks, EV demand boom, recession, currency moves) that show ranges.

**Result**: Guidance becomes ranges: EBIT £1.2B–£1.6B under base assumptions.

**File**: `config/settings.yaml` (7 scenarios defined), `src/drivers/scenario_builder.py`

---

### 5. **Explainability-First Design**
**Problem**: Why did the model say Lithium would go up? Forecasts with no explanation lose board trust.

**Solution**: Track feature importance (macro factors, seasonality, recent shocks), generate narrative explanations.

**File**: `src/governance/explainability_engine.py`, `src/governance/audit_logger.py`

---

## Data Flow: From Real Prices to P&L Distribution

```
Step 1: Fetch Data (daily/weekly)
────────────────────────────────
    fetch_data.py → Yahoo Finance + FRED API
    └─ Saves to: data/raw/commodity_prices.csv (7 years, 12 commodities)
                 data/raw/macro_indicators.csv (macro context)

Step 2: Train Models (one-time, ~5 min)
──────────────────────────────────────
    run_commodity_pipeline.py:
    ├─ Stage 1: Load + clean (Polars)
    ├─ Stage 2: Feature engineering (lags, momenta, macro context)
    ├─ Stage 3: Train SARIMAX (seasonal decomposition)
    ├─ Stage 4: Train XGBoost (50+ features, 5-fold CV)
    ├─ Stage 5: Regime detection (Hurst exponent on residuals)
    ├─ Stage 6: Generate forecasts (12-month forward)
    └─ Stage 7: Validate backtest, save models to models/saved/
    
    └─ Output: models/pipeline_results.json (CV metrics for each commodity)

Step 3: Generate Report or Dashboard (on-demand)
───────────────────────────────────────────────
    generate_executive_report.py:
    ├─ Load trained models + latest real data
    ├─ Compute commodity statistics (mean, vol, YoY)
    ├─ Evaluate accuracy benchmarks (walk-forward CV)
    ├─ Project 12-month forecasts
    ├─ Load real CV metrics from pipeline_results.json
    ├─ Build 7 scenarios (commodity ± shocks, demand shifts)
    ├─ Project P&L under each scenario
    └─ Generate 766-line markdown report
    
    OR
    
    streamlit run src/dashboard/app.py:
    ├─ Load data + models (lazy caching)
    ├─ Render 8 interactive pages
    ├─ Live shock calculator (move commodity slider → EBIT updates)
    └─ Export graphs, download report

Step 4: Risk Quantification (run_full_architecture.py)
──────────────────────────────────────────────────────
    ├─ Layer 3: Compute deterministic P&L (base case)
    ├─ Layer 4: Run 10,000 Monte Carlo simulations
    │   ├─ Sample commodity returns (multivariate normal + fat tail)
    │   ├─ Sample demand shocks
    │   └─ Propagate through financial model
    ├─ Compute EBIT distribution:
    │   ├─ Mean: £1,360M
    │   ├─ 80% CI: £1,231M – £1,571M
    │   ├─ VaR(95%): £705M downside
    │   └─ Skewness: -0.15 (slight left tail)
    └─ Output: Risk metrics for dashboard + report

Done! Now your CFO has:
├─ Point forecasts (base case)
├─ Uncertainty ranges (80% CI)
├─ Directional signals (upcoming trends)
├─ Optimal hedge ratios (save £1.5M/yr)
└─ Explainability ("Why up? Because Chinese EV demand is recovering")
```

> **Want the actual numbers?** See [docs/FULL_ARCHITECTURE_RUN.md](FULL_ARCHITECTURE_RUN.md) for the complete 2024 backtest trace — real commodity data, actual MAPE per commodity, exact P&L by segment, hedge savings, and Monte Carlo calibration proof.

---

## Module Reference

### `src/data/` — Data Layer
- **`polars_pipeline.py`**: CSV → Parquet conversion, lazy evaluation
- **`data_source_protocol.py`**: Abstract interfaces (Protocol patterns)
- **`synthetic_generator.py`**: JLR-calibrated data (O-U processes, realistic correlations)
- **`connectors/`**: Yahoo Finance, FRED, CCXT, SAP stubs

**How to swap data sources:**
1. Edit `config/settings.yaml`: `data_source: synthetic` → `data_source: parquet`
2. Zero code changes downstream (Protocol abstraction handles it)

---

### `src/models/` — ML Layer
- **`commodity_forecast.py`**: SARIMAX + XGBoost ensemble with regime weighting
- **`regime_detector.py`**: Hurst exponent for trending vs. mean-reverting classification
- **`demand_forecast.py`**: Per-segment volume forecasting (XGBoost)
- **`price_elasticity.py`**: Log-log regression (own-price + cross-commodity)

**Key metric:** CV MAPE (cross-validation mean absolute percentage error) — the most honest accuracy estimate.

---

### `src/drivers/` — Financial Model
- **`financial_model.py`**: Revenue, COGS, margin calculations
- **`scenario_builder.py`**: 7 pre-cooked scenarios (commodity ±20%, demand ±10%, etc.)
- **`shock_injector.py`**: Insert "what-if" commodity prices → recalc P&L

**Key output:** Monthly P&L projections + sensitivity tables.

---

### `src/simulation/` — Risk Layer
- **`monte_carlo.py`**: 10,000-run simulation with fat-tail distribution
- **`hedge_optimizer.py`**: Portfolio theory optimization (minimize EBIT variance)

**Key output:** VaR, CVaR, margin distribution, optimal hedge ratios.

---

### `src/dashboard/` — UI Layer
- **`app.py`**: Streamlit multi-page app (8 pages)
- **`pages/`**: Individual page renderers
- **`helpers.py`**: Common charts, caching, formatting

**Key feature:** Real-time interactivity — move a commodity slider and EBIT recalculates in <1 second.

---

### `src/governance/` — Explainability & Audit
- **`audit_logger.py`**: Immutable JSONL trail of every forecast, override, and decision
- **`explainability_engine.py`**: Feature importance + narrative explanations
- **`market_intelligence.py`**: Generates alert signals (regime shifts, unusual moves)

---

## Configuration Reference

**File:** `config/settings.yaml`

```yaml
# Data source: synthetic, parquet, sap, salesforce
data_source: synthetic

# BOM weights (all 12 commodities)
bom_weights:
  Steel: 0.22
  Lithium: 0.18
  Aluminum: 0.12
  Cobalt: 0.07
  ... (12 total, must sum to 1.0)

# Forecast horizon (months)
forecast_horizon_months: 12

# Monte Carlo parameters
monte_carlo_runs: 10000
confidence_level: 0.80

# 7 Scenarios (pre-defined)
scenarios:
  - base: {lithium: 1.0, aluminum: 1.0, demand: 1.0}
  - bear: {lithium: 1.2, aluminum: 1.15, demand: 0.95}
  - bull: {lithium: 0.8, aluminum: 0.85, demand: 1.05}
  ... (4 more)
```

---

## Testing & Validation

**Run all tests:**
```bash
pytest tests/ -v
```

**What's tested:**
- Data pipeline correctness (Polars operations)
- Model accuracy (backtests vs. benchmarks)
- Financial model math (COGS, margin calculations)
- Monte Carlo calibration (CI coverage)
- Dashboard rendering (Streamlit component tests)

**All 34 tests pass** — indicates no regressions.

---

## Production Deployment

Current state: **POC on synthetic + real market data**

Production readiness (planned):
1. **Data layer**: Replace `synthetic_generator.py` with actual SAP connector
2. **Model layer**: Add real warehouse data, retrain daily
3. **Financial layer**: Integrate actual general ledger COGS actuals
4. **Simulation layer**: Add risk model approved by CFO
5. **Governance**: Audit trail feeds into quarterly financial review

---

## Troubleshooting Guide

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Dashboard slow | Models not cached, reloading data | Restart dashboard, check data/external/ for cached parquets |
| Forecast seems wrong | Wrong regime classification | Check regime_detector output, inspect Hurst exponent |
| Monte Carlo crash | Singular covariance matrix | Check correlation data, ensure no duplicate columns |
| EBIT doesn't match | COGS or revenue driver error | Validate BOM weights sum to 1.0 in config |

---

## Further Reading

| Document | What You'll Find |
|----------|------------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, first run, common tasks (5 min) |
| [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) | How to interpret every number in the report and dashboard |
| [FULL_ARCHITECTURE_RUN.md](FULL_ARCHITECTURE_RUN.md) | Actual 2024 backtest trace with real MAPE, P&L by segment, MC calibration |
| [EXECUTIVE_INTELLIGENCE_REPORT.md](EXECUTIVE_INTELLIGENCE_REPORT.md) | Latest board-ready intelligence report (auto-generated daily) |
| [../TECHNICAL_ASSESSMENT.md](../TECHNICAL_ASSESSMENT.md) | What's production-ready, known gaps, Q3–Q4 roadmap |
