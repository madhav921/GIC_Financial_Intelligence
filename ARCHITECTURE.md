# GIC Plan-to-Perform — Architecture Reference

> **Single-view architecture guide.** Open this file first.

---

## 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        orchestrator.py                                  │
│                        GICOrchestrator                                  │
│              engine.run_full_pipeline() / engine.quick_pnl()            │
└────────┬──────────┬──────────┬──────────┬──────────────────────────────┘
         │          │          │          │
         ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Governance & LLM             layers/layer5_governance/       │
│                                                                         │
│  GovernanceLayerController                                              │
│  ├── GICLLMEngine           (Ollama / HuggingFace / template)           │
│  ├── AuditTrail             (append-only JSONL, UUID + ISO8601)         │
│  ├── BiasTracker            (>5% alert, >10% escalation)                │
│  └── ExplainabilityEngine   (XGBoost top-driver narratives)             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ audit + narratives
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 4 — Simulation & Risk            layers/layer4_simulation/       │
│                                                                         │
│  SimulationLayerController                                              │
│  ├── MonteCarloEngine       (10K sims, Normal + t(df=5) distributions)  │
│  ├── ScenarioEngine         (7 preset: Base/Bull/Bear/Crisis/...)       │
│  └── HedgeOptimizer         (portfolio-theory hedge ratios)             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ VaR, CVaR, fan chart
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — Financial Drivers            layers/layer3_financial/        │
│                                                                         │
│  FinancialLayerController                                               │
│  ├── FinancialModel         (P&L assembly engine)                       │
│  ├── RevenueDrivers         (Volume × Net Price × (1 - Incentive%))     │
│  ├── CostDrivers            (BOM-weighted COGS + commodity index)       │
│  └── CapitalDrivers         (CapEx scheduling, straight-line depr.)     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ monthly P&L
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Predictive Intelligence      layers/layer2_intelligence/     │
│                                                                         │
│  IntelligenceLayerController                                            │
│  ├── CommodityForecastModel (SARIMAX + XGBoost ensemble)                │
│  ├── RegimeDetector         (Hurst exponent adaptive weighting)         │
│  ├── CommodityScenarios     (Bear/Base/Bull hybrid)                     │
│  ├── FuturesCurve           (CME/NYMEX term structure)                  │
│  └── DemandForecast         (XGBoost per segment)                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ forecasts, commodity index
┌────────────────────────────────▼────────────────────────────────────────┐
│  LAYER 1 — Data Architecture            layers/layer1_data/             │
│                                                                         │
│  DataLayerController                                                    │
│  ├── YFinanceConnector      (9 commodities, equity indices, FX)         │
│  ├── FREDConnector          (PMI, CPI, Fed Funds, yield curve)          │
│  ├── CCXTConnector          (Binance crypto OHLCV)                      │
│  ├── PolarsPipeline         (Parquet columnar storage)                  │
│  ├── SyntheticGenerator     (Ornstein-Uhlenbeck fallback)               │
│  ├── FeatureEngineering     (lags, rolling, RSI, MACD, calendar)        │
│  └── DataRouter             (Parquet → CSV → Synthetic priority chain)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
External Sources                  Synthetic Fallback
  yfinance (Yahoo)                  O-U mean-reverting
  FRED (macro)            →  L1  ←  process (always
  CCXT (Binance)                    available)
  CME/NYMEX futures

        │ commodity_df, macro_df, sales_df, bom_df
        ▼
  ┌─── L2 ───────────────────────────────────────────┐
  │  Regime detection  →  SARIMAX / XGBoost / Scenario│
  │  Hurst exponent    →  ensemble weight adaptation  │
  └──────────────────────────────────────────────────┘
        │ forecasts{}, commodity_index_df
        ▼
  ┌─── L3 ───────────────────────────────────────────┐
  │  Revenue  = Σ(Vol × Price × (1-Incentive%))       │
  │  COGS     = Rev × 77.5% × commodity_multiplier    │
  │  EBIT     = Gross Margin - Warranty - Depr.        │
  └──────────────────────────────────────────────────┘
        │ pnl_df (monthly), annual_df
        ▼
  ┌─── L4 ───────────────────────────────────────────┐
  │  Monte Carlo 10K sims   →  VaR(95%), CVaR(95%)    │
  │  7 preset scenarios     →  comparison table        │
  │  Variance decomposition →  commodity/demand/FX %  │
  └──────────────────────────────────────────────────┘
        │ SimulationResult, scenario_comparison
        ▼
  ┌─── L5 ───────────────────────────────────────────┐
  │  JSONL audit trail      →  immutable compliance   │
  │  LLM narratives         →  Ollama / flan-t5 / tmpl│
  │  Bias tracking          →  >5% alert / >10% esc.  │
  │  Explainability         →  top feature drivers    │
  └──────────────────────────────────────────────────┘
        │ results dict
        ▼
  Dashboard / API / Report
```

---

## Layer-by-Layer Reference

### Layer 1: Data Architecture

| Property | Value |
|---|---|
| Folder | `layers/layer1_data/` |
| Controller | `DataLayerController` |
| src/ modules | `src/data/` |

**Primary API:**

```python
data = DataLayerController()
commodity_df = data.load_commodity_data()   # Real yfinance → Parquet → CSV → synthetic
macro_df     = data.load_macro_data()       # FRED → synthetic
sales_df     = data.load_sales_data()       # Synthetic vehicle segment volumes
bom_df       = data.load_bom_data()         # Bill of Materials commodity weights
features_df  = data.build_feature_matrix(commodity_df, macro_df, "Copper")

# One-shot loader (recommended):
commodity_df, macro_df, sales_df, bom_df = data.load_all()
```

**Source priority chain:**
```
Parquet (fastest, external/)
  → Real CSV (raw/)
  → Synthetic CSV (synthetic/)
  → Generated O-U synthetic (SyntheticDataGenerator)
```

**Key modules:**

| Module | Class/Function | Purpose |
|---|---|---|
| `src/data/connectors/yfinance_connector.py` | `YFinanceConnector` | 9 commodities + indices + FX |
| `src/data/connectors/fred_connector.py` | `FREDConnector` | PMI, CPI, Fed Funds, 10Y yield |
| `src/data/connectors/ccxt_connector.py` | `CCXTConnector` | Binance crypto OHLCV |
| `src/data/polars_pipeline.py` | `PolarsPipeline` | Parquet read/write, lazy evaluation |
| `src/data/synthetic_generator.py` | `SyntheticDataGenerator` | O-U mean-reverting prices |
| `src/data/feature_engineering.py` | `prepare_commodity_features()` | Lag, rolling, RSI, MACD, calendar |
| `src/data/data_router.py` | `DataRouter` | Source-agnostic routing protocol |

---

### Layer 2: Predictive Intelligence

| Property | Value |
|---|---|
| Folder | `layers/layer2_intelligence/` |
| Controller | `IntelligenceLayerController` |
| src/ modules | `src/models/` |

**Primary API:**

```python
intel = IntelligenceLayerController()
metrics        = intel.train_all_models(commodity_df, macro_df)    # SARIMAX + XGBoost, 5-fold CV
result         = intel.forecast_commodity("Copper", commodity_df)  # Ensemble forecast
all_forecasts  = intel.forecast_all_commodities(commodity_df)      # All trained commodities
index_df       = intel.generate_commodity_index(commodity_df)      # BOM-weighted index (base=100)
regime         = intel.detect_regime(price_series)                 # Hurst exponent classification
importance_df  = intel.get_feature_importance("Copper")            # XGBoost top features
cv_df          = intel.get_cv_metrics()                            # Walk-forward MAPE table
```

**Forecast method routing (Hurst exponent H):**

```
H < 0.45  →  Mean-reverting  →  SARIMAX(1,1,1)(1,1,1,12) dominant (weight 0.6)
H > 0.55  →  Trending        →  XGBoost + 50 macro features dominant (weight 0.6)
H ≈ 0.50  →  Volatile        →  Scenario (Bear/Base/Bull) dominant (weight 0.5)
```

**Key modules:**

| Module | Class | Purpose |
|---|---|---|
| `src/models/commodity_forecast.py` | `CommodityForecastModel` | Training orchestrator, ensemble router |
| `src/models/regime_detector.py` | `RegimeDetector` | Hurst exponent + regime classification |
| `src/models/commodity_scenarios.py` | `CommodityScenarioModel` | Bear/Base/Bull with macro shifting |
| `src/models/futures_curve.py` | `FuturesCurveModel` | CME/NYMEX term structure |
| `src/models/demand_forecast.py` | `DemandForecastModel` | XGBoost per vehicle segment |
| `src/models/price_elasticity.py` | `PriceElasticityModel` | Log-log Ridge regression |
| `src/models/model_registry.py` | `ModelRegistry` | Versioned joblib persistence |
| `src/models/backtesting.py` | `WalkForwardValidator` | 5-fold, no look-ahead bias |

---

### Layer 3: Financial Drivers

| Property | Value |
|---|---|
| Folder | `layers/layer3_financial/` |
| Controller | `FinancialLayerController` |
| src/ modules | `src/drivers/` |

**Primary API:**

```python
fin = FinancialLayerController()
pnl_df   = fin.build_pnl(sales_df, commodity_index_df)               # Monthly P&L
annual   = fin.annual_summary(pnl_df)                                # Annual rollup
scenario = fin.apply_scenario(sales_df, commodity_index_df,
                               demand_shock=-0.08, commodity_shock=0.40)
shocked  = fin.apply_commodity_shock(pnl_df, {"Lithium": 0.20})      # <1 second
```

**P&L output columns:**
```
date | segment | volume | net_revenue | total_cogs | gross_margin |
warranty_reserve | depreciation | operating_income | operating_margin_pct |
tax | net_income
```

**Financial equations:**

```
Revenue          = Σ(Volume_seg × Net_Price_seg × (1 - Incentive%_seg))
COGS             = Revenue × 77.5% × (1 + Commodity_Impact × 45%)
Gross Margin     = Revenue - COGS
Warranty Reserve = Revenue × 2.5%
Depreciation     = CapEx / Useful_Life / 12
Operating Income = Gross Margin - Warranty Reserve - Depreciation
Tax              = max(0, Operating_Income × 21%)
Net Income       = Operating_Income - Tax
```

**Key modules:**

| Module | Class | Purpose |
|---|---|---|
| `src/drivers/financial_model.py` | `FinancialModel` | P&L assembly, scenario builder |
| `src/drivers/revenue_drivers.py` | `RevenueDriver` | Volume × price with elasticity |
| `src/drivers/cost_drivers.py` | `CostDriver` | BOM-weighted COGS + commodity index |
| `src/drivers/capital_drivers.py` | `CapitalDriver` | CapEx schedule, depreciation |

---

### Layer 4: Simulation & Risk

| Property | Value |
|---|---|
| Folder | `layers/layer4_simulation/` |
| Controller | `SimulationLayerController` |
| src/ modules | `src/simulation/`, `src/models/hedge_optimizer.py` |

**Primary API:**

```python
sim = SimulationLayerController()
mc_result    = sim.run_monte_carlo(sales_df, commodity_index_df, n_simulations=10_000)
scenarios    = sim.run_all_scenarios(sales_df, commodity_index_df)   # 7 preset scenarios
comparison   = sim.compare_scenarios(scenarios)                      # Summary table
fan_df       = sim.run_monthly_fan(base_pnl)                        # P5/P25/P75/P95 bands
risk         = sim.decompose_risk(sales_df, commodity_index_df)     # {commodity_pct, demand_pct, fx_pct}
hedge        = sim.optimize_hedge("Copper", exposure, spot, mean, std)
```

**Shock distributions:**

```
Demand:    Normal(μ=scenario_mean, σ=10%)
Commodity: Student's t(df=5, μ=scenario_mean, σ=20%)  — fat tails
FX:        Normal(μ=0, σ=5%)
```

**7 preset scenarios:**

| Scenario | Demand Shock | Commodity Shock |
|---|---|---|
| Base | 0% | 0% |
| Bull | +8% | -10% |
| Bear | -8% | +20% |
| Commodity Crisis | -5% | +40% |
| Lithium +15% | 0% | +15% (Lithium only) |
| EU Demand -8% | -8% | 0% |
| Stagflation | -12% | +35% |

**Validated calibration (2024 backtest):**
- 80% CI coverage: 79% (target ≥ 80%) — near-perfect
- VaR(95%): £705M downside
- CVaR(95%): £-452M average worst-5%

**Key modules:**

| Module | Class | Purpose |
|---|---|---|
| `src/simulation/monte_carlo.py` | `MonteCarloEngine` | 10K–50K sim engine, VaR/CVaR |
| `src/simulation/scenario_engine.py` | `ScenarioEngine` | 7 preset + custom scenario runner |
| `src/models/hedge_optimizer.py` | `HedgeOptimizer` | Portfolio-theory hedge ratios |

---

### Layer 5: Governance & LLM

| Property | Value |
|---|---|
| Folder | `layers/layer5_governance/` |
| Controller | `GovernanceLayerController` |
| LLM Engine | `GICLLMEngine` (`llm_engine.py`) |
| src/ modules | `src/governance/` |

**Primary API:**

```python
gov = GovernanceLayerController()

# Audit trail
event_id = gov.log_event("forecast_generated", {"commodity": "Copper", "mape": 7.0})
history  = gov.get_audit_history(limit=100)

# LLM narratives (auto-detects backend)
narrative        = gov.generate_narrative(forecast_result)
risk_narrative   = gov.generate_risk_narrative(scenario_name, sim_result, risk_decomp)
exec_insight     = gov.generate_executive_insight(pnl_summary, commodity_index=102.3)

# Bias tracking
bias_report = gov.track_bias("Copper", actual_prices, forecast_prices)
#  → {bias_pct, mae, mape, alert_triggered, escalation_required}

# Explainability
explanation = gov.explain_forecast("Copper", forecast_result)
#  → {commodity, top_drivers, narrative, llm_narrative, forecast_value}

# LLM health
status = gov.llm_health_check()
#  → {backend, model, status}
```

**LLM backend auto-detection:**

```
Priority 1: Ollama (llama3.2:1b)
  → Requires: ollama pull llama3.2:1b
  → Best narrative quality

Priority 2: HuggingFace transformers (google/flan-t5-base)
  → pip install "gic-plan-to-perform[llm]"
  → ~300MB download on first run, CPU-compatible

Priority 3: Template-based fallback
  → Always available, zero dependencies
  → Context-aware templates for forecasts, alerts, risk, executive summaries
```

**Audit trail JSONL format:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-06-11T09:15:32.441Z",
  "event_type": "forecast_generated",
  "details": {"commodity": "Copper", "mape": 7.2, "model_type": "ensemble"},
  "user": "system"
}
```

**Bias tracking thresholds:**

```
variance ≤ 5%   →  Normal — no action
5% < variance ≤ 10%  →  Alert logged to audit trail
variance > 10%  →  Escalation event + LLM narrative + CFO notification
```

**Key modules:**

| Module | Class | Purpose |
|---|---|---|
| `layers/layer5_governance/llm_engine.py` | `GICLLMEngine` | Open-source LLM (Ollama/HF/template) |
| `src/governance/audit_trail.py` | `AuditTrail` | Append-only JSONL with UUID+ISO8601 |
| `src/governance/bias_tracking.py` | `BiasTracker` | Residual monitoring + alert thresholds |
| `src/governance/explainability.py` | `ExplainabilityEngine` | Feature importance → narrative |

---

## File Map

```
GIC_Financial_Intelligence/
│
├── orchestrator.py                    ← ROOT ENTRY POINT (GICOrchestrator)
├── ARCHITECTURE.md                    ← This file
├── pyproject.toml                     ← Dependencies (+ [llm] and [llm-ollama] extras)
│
├── layers/                            ← 5-Layer architecture (wraps src/)
│   ├── __init__.py                    ← Exports all 5 controllers
│   │
│   ├── layer1_data/
│   │   ├── __init__.py                ← Layer 1 docstring + DataLayerController export
│   │   ├── controller.py              ← DataLayerController (load_all, load_commodity, ...)
│   │   └── README.md                  ← 10-line developer orientation
│   │
│   ├── layer2_intelligence/
│   │   ├── __init__.py                ← Layer 2 docstring + IntelligenceLayerController export
│   │   ├── controller.py              ← IntelligenceLayerController (train, forecast, regime)
│   │   └── README.md
│   │
│   ├── layer3_financial/
│   │   ├── __init__.py                ← Layer 3 docstring + FinancialLayerController export
│   │   ├── controller.py              ← FinancialLayerController (build_pnl, apply_scenario)
│   │   └── README.md
│   │
│   ├── layer4_simulation/
│   │   ├── __init__.py                ← Layer 4 docstring + SimulationLayerController export
│   │   ├── controller.py              ← SimulationLayerController (run_monte_carlo, ...)
│   │   └── README.md
│   │
│   └── layer5_governance/
│       ├── __init__.py                ← Layer 5 docstring + exports
│       ├── controller.py              ← GovernanceLayerController (log, narrate, track)
│       ├── llm_engine.py              ← GICLLMEngine (Ollama/HuggingFace/template)
│       └── README.md
│
├── src/                               ← Core implementation (DO NOT DELETE)
│   ├── config.py
│   ├── data/
│   │   ├── connectors/
│   │   │   ├── yfinance_connector.py  ← Yahoo Finance fetcher
│   │   │   ├── fred_connector.py      ← FRED macro fetcher
│   │   │   ├── ccxt_connector.py      ← Binance crypto fetcher
│   │   │   ├── commodity_api.py
│   │   │   ├── data_lake.py
│   │   │   └── erp_connector.py
│   │   ├── polars_pipeline.py         ← Parquet columnar storage
│   │   ├── synthetic_generator.py     ← O-U synthetic data
│   │   ├── feature_engineering.py     ← ML feature builder
│   │   ├── data_router.py             ← Source-agnostic routing
│   │   └── data_loader.py
│   │
│   ├── models/
│   │   ├── commodity_forecast.py      ← CommodityForecastModel (primary L2 engine)
│   │   ├── regime_detector.py         ← Hurst exponent
│   │   ├── commodity_scenarios.py     ← Bear/Base/Bull
│   │   ├── futures_curve.py           ← CME/NYMEX term structure
│   │   ├── demand_forecast.py         ← XGBoost demand
│   │   ├── price_elasticity.py        ← Ridge log-log
│   │   ├── model_registry.py          ← joblib persistence
│   │   ├── backtesting.py             ← Walk-forward CV
│   │   ├── hedge_optimizer.py         ← Portfolio hedge ratios
│   │   ├── commodity_forecast_xgboost.py
│   │   ├── commodity_shock.py
│   │   └── inventory_risk.py
│   │
│   ├── drivers/
│   │   ├── financial_model.py         ← P&L assembly engine
│   │   ├── revenue_drivers.py         ← Volume × Price
│   │   ├── cost_drivers.py            ← BOM-weighted COGS
│   │   └── capital_drivers.py         ← CapEx / depreciation
│   │
│   ├── simulation/
│   │   ├── monte_carlo.py             ← MC engine (10K sims)
│   │   └── scenario_engine.py         ← 7 preset scenarios
│   │
│   ├── governance/
│   │   ├── audit_trail.py             ← JSONL audit log
│   │   ├── bias_tracking.py           ← Forecast residuals
│   │   └── explainability.py          ← Feature importance
│   │
│   ├── analytics/
│   │   ├── ffn_analytics.py
│   │   └── market_intelligence.py
│   │
│   ├── api/
│   │   ├── app.py                     ← FastAPI application
│   │   └── routes/
│   │       ├── forecast.py
│   │       ├── pnl.py
│   │       ├── simulation.py
│   │       └── health.py
│   │
│   └── dashboard/
│       ├── app.py                     ← Streamlit entry point
│       └── pages/
│           ├── executive_summary.py
│           ├── commodity_intelligence.py
│           ├── financial_pnl.py
│           ├── scenario_simulation.py
│           ├── market_monitor.py
│           ├── intelligence_report.py
│           ├── data_explorer.py
│           └── backtesting.py
│
├── data/
│   └── synthetic/                     ← Pre-generated fallback CSVs
│       ├── commodity_prices.csv
│       ├── macro_indicators.csv
│       ├── sales_data.csv
│       ├── production_inventory.csv
│       └── bom_data.csv
│
└── scripts/                           ← Standalone runners
    ├── run_full_architecture.py
    ├── run_pipeline.py
    ├── run_commodity_pipeline.py
    ├── fetch_data.py
    ├── generate_data.py
    ├── train_models.py
    └── generate_executive_report.py
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Run the full pipeline
python orchestrator.py

# 3. Quick P&L scenario (no model training)
python -c "
from orchestrator import GICOrchestrator
engine = GICOrchestrator()
print(engine.quick_pnl(demand_shock=-0.08, commodity_shock=0.40))
"

# 4. Use individual layers directly
python -c "
from layers import DataLayerController, IntelligenceLayerController
data = DataLayerController()
commodity_df, macro_df, sales_df, bom_df = data.load_all()
print(f'Loaded {len(commodity_df)} months of commodity data')
"

# 5. Enable LLM narratives (HuggingFace, free, ~300MB)
pip install -e ".[llm]"
python -c "
from layers.layer5_governance.llm_engine import GICLLMEngine
llm = GICLLMEngine()
print(llm.explain_forecast('Copper', forecast_pct=7.2, drivers=['PMI', 'DXY', 'inventory']))
"

# 6. Enable Ollama (best quality, local, requires Ollama installed)
pip install -e ".[llm-ollama]"
ollama pull llama3.2:1b
# LLM engine auto-detects and uses Ollama on next run

# 7. Launch the Streamlit dashboard
streamlit run src/dashboard/app.py

# 8. Launch the FastAPI server
uvicorn src.api.app:app --reload
```

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Columnar data pipeline | Polars + PyArrow | >=1.0.0 |
| Tabular data | Pandas | >=2.0.0 |
| Time-series forecasting | statsmodels SARIMAX | >=0.14.0 |
| Gradient boosting | XGBoost | >=2.0.0 |
| ML utilities | scikit-learn | >=1.3.0 |
| Open-source LLM (small) | HuggingFace transformers flan-t5-base | >=4.36.0 |
| Open-source LLM (large) | Ollama llama3.2:1b | latest |
| Financial data | yfinance | >=0.2.36 |
| Macro data | fredapi | >=0.5.1 |
| Crypto data | ccxt | >=4.2.0 |
| Dashboard | Streamlit | >=1.30.0 |
| REST API | FastAPI + Uvicorn | >=0.104.0 |
| Visualisation | Plotly | >=5.18.0 |
| Logging | loguru | >=0.7.0 |
| Model persistence | joblib | >=1.3.0 |
| HTTP client (Ollama) | httpx | >=0.25.0 |
| Configuration | pydantic-settings + PyYAML | >=2.1.0 |

---

## Design Principles

1. **Layer independence** — Each layer has a single `controller.py` entry point.
   Layers communicate only through DataFrames and plain Python dicts/dataclasses.
   No layer imports from another layer except via `orchestrator.py`.

2. **src/ is unchanged** — `layers/` wraps `src/` without modification.
   All business logic stays in `src/`. Controllers are thin delegation shells.

3. **Graceful degradation** — Every external dependency has a fallback:
   - Real market data → synthetic O-U data
   - Ollama LLM → HuggingFace flan-t5 → template narratives
   - Parquet files → CSV → synthetic generation

4. **Immutable audit trail** — Layer 5 appends to JSONL; never edits or deletes.
   Every pipeline run is fully traceable: data source, model version, scenario parameters.

5. **Sub-second shock recalculation** — Layer 3 commodity shock injection is vectorised
   over a pre-built P&L DataFrame. No model re-training required.
