# GIC Plan-to-Perform Engine

### AI-Powered Commodity Forecast & Financial Intelligence Platform

> **Enterprise-grade financial planning engine** that combines real-time market data, ML-driven commodity forecasting, deterministic financial modeling, and Monte Carlo simulation — built for CFO-level strategic decision making.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/data-Polars%20%2B%20Parquet-orange.svg)](https://pola.rs)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-34%20passed-green.svg)](#testing)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE DATA INPUTS (Layer 1)                        │
│                                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Yahoo       │  │   CCXT       │  │  FRED        │  │  Synthetic         │  │
│  │  Finance     │  │  (Binance)   │  │  (Macro)     │  │  Generator         │  │
│  │              │  │              │  │              │  │                    │  │
│  │ • Commodities│  │ • BTC/ETH    │  │ • Fed Rate   │  │ • O-U Process      │  │
│  │ • FX Rates   │  │ • SOL/XRP    │  │ • CPI/PPI    │  │ • 8 Commodities    │  │
│  │ • S&P/VIX    │  │ • BNB/AVAX   │  │ • GDP/Unemp  │  │ • 4 Segments       │  │
│  │ • Oil/Gold   │  │ • 6 Pairs    │  │ • Sentiment  │  │ • BOM/Inventory    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘  │
│         │                 │                 │                    │              │
│         └────────┬────────┴────────┬────────┘                    │              │
│                  │  Polars + Parquet Pipeline                    │              │
│                  └──────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PREDICTIVE INTELLIGENCE LAYER (Layer 2)                       │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────────────┐  │
│  │ Commodity      │   │ Demand         │   │ Price Elasticity              │  │
│  │ Forecast       │   │ Forecast       │   │ Model                        │  │
│  │ (12 Materials) │   │                │   │                              │  │
│  │ SARIMAX+XGB    │──▶│ XGBoost per    │──▶│ Log-Log Ridge Regression     │  │
│  │ Futures+Scenar │   │ Segment        │   │ Own-price + Commodity cross  │  │
│  │ 4 Methods      │   │ (4 segments)   │   │ elasticity estimation        │  │
│  └───────┬────────┘   └───────┬────────┘   └───────────────┬──────────────┘  │
│          │                    │                             │                  │
│  ┌───────▼────────┐   ┌──────▼─────────┐   ┌──────────────▼──────────────┐  │
│  │ Commodity      │   │ Volume(t)      │   │ Inventory & Warranty       │  │
│  │ Index(t)       │   │ Forecasts      │   │ Risk Models                │  │
│  └───────┬────────┘   └───────┬────────┘   └───────────────┬────────────┘  │
│          │                    │                             │                │
│  ┌───────▼────────────────────▼─────────────────────────────▼──────────────┐  │
│  │                    FFN Performance Analytics                            │  │
│  │         CAGR • Sharpe • Sortino • Max Drawdown • Calmar                │  │
│  │         Rolling Volatility • Return Correlation Matrix                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│               DETERMINISTIC FINANCIAL DRIVER MODEL (Layer 3)                    │
│                                                                                 │
│     Revenue = Volume × Net Price          Margin = Revenue − COGS              │
│     COGS = f(BOM, Commodity Index)        Inventory = Production − Sales       │
│                                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ Revenue        │  │ Cost           │  │ Capital        │  │ Full P&L     │  │
│  │ Drivers        │  │ Drivers        │  │ Drivers        │  │ Engine       │  │
│  │                │  │                │  │                │  │              │  │
│  │ Vol × Price    │  │ Material 45%   │  │ Depreciation   │  │ Revenue      │  │
│  │ Incentives     │──▶│ Labor 30%     │──▶│ CapEx Plans   │──▶│ − COGS      │  │
│  │ Demand Shock   │  │ Commodity Idx  │  │ Useful Life    │  │ − Warranty   │  │
│  │ scenarios      │  │ FX exposure    │  │ schedules      │  │ − Deprec.    │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  │ = Net Income │  │
│                                                               └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│              SCENARIO SIMULATION & BUSINESS DECISIONS (Layer 4)                  │
│                                                                                 │
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────────────────┐  │
│  │ Scenario Inputs  │  │ Monte Carlo       │  │ Outputs                     │  │
│  │                  │  │ Engine            │  │                             │  │
│  │ • Demand shock   │  │                   │  │ • Margin Impact Dist.       │  │
│  │ • Commodity +40% │──▶│ 10,000 sims      │──▶│ • VaR / CVaR (95%)        │  │
│  │ • FX movements   │  │ t-distribution    │  │ • Cash Flow Risk           │  │
│  │ • 7 Presets      │  │ (fat tails, df=5) │  │ • Percentile Analysis      │  │
│  └──────────────────┘  └───────────────────┘  │ • Strategic Planning       │  │
│                                                └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GOVERNANCE LAYER (Layer 5)                               │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐   │
│  │ Audit Trail   │  │ Bias Tracking │  │ Explainability│  │ Market         │   │
│  │ (JSONL)       │  │               │  │               │  │ Intelligence   │   │
│  │               │  │ Mean/Median   │  │ Natural-lang  │  │                │   │
│  │ Append-only   │  │ Bias metrics  │  │ explanations  │  │ Regime detect. │   │
│  │ UUID entries  │  │ Trend alerts  │  │ Model cards   │  │ Alert engine   │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                 ┌────────────┐ ┌──────────┐ ┌────────────┐
                 │ Streamlit  │ │ FastAPI   │ │ Executive  │
                 │ Dashboard  │ │ REST API  │ │ Reports    │
                 │ (6 pages)  │ │ + Swagger │ │            │
                 └────────────┘ └──────────┘ └────────────┘
```

---

## Project Structure

```
gic-plan-to-perform/
│
├── config/
│   └── settings.yaml              # Master configuration (commodities, segments, thresholds)
│
├── data/
│   ├── external/                  # Real-world data (Parquet) — Yahoo Finance, CCXT, FRED
│   ├── parquet/                   # Cached Parquet files (auto-converted from CSV)
│   ├── processed/                 # Transformed feature sets
│   ├── raw/                       # Raw ingested data
│   └── synthetic/                 # Generated test data (CSV)
│
├── models/
│   └── saved/                     # Trained model artifacts (joblib + metadata JSON)
│
├── logs/
│   └── audit/                     # Governance audit trail (JSONL)
│
├── scripts/
│   ├── fetch_data.py              # Fetch real-world data from all sources
│   ├── generate_data.py           # Generate synthetic data for testing
│   ├── train_models.py            # Train all ML models
│   ├── run_pipeline.py            # End-to-end pipeline orchestration
│   └── run_commodity_pipeline.py  # 8-stage commodity forecast pipeline
│
├── src/
│   ├── config.py                  # Configuration loader (YAML + env vars)
│   ├── logging_setup.py           # Loguru logging configuration
│   │
│   ├── analytics/                 # Financial analytics & market intelligence
│   │   ├── ffn_analytics.py       # FFN: CAGR, Sharpe, drawdowns, correlations
│   │   └── market_intelligence.py # Regime detection, alerts, risk scoring
│   │
│   ├── api/                       # FastAPI REST service
│   │   ├── app.py                 # Application factory + CORS + lifespan
│   │   ├── schemas.py             # Pydantic v2 request/response models
│   │   └── routes/
│   │       ├── forecast.py        # /forecast/commodity, /forecast/commodity-index
│   │       ├── health.py          # /health, /models
│   │       └── simulation.py      # /simulation/scenario, /simulation/presets
│   │
│   ├── dashboard/                 # Streamlit multi-page dashboard
│   │   ├── app.py                 # Dashboard entry point
│   │   ├── helpers.py             # Shared utilities (caching, formatting)
│   │   └── pages/
│   │       ├── executive_summary.py      # CFO overview — KPIs, alerts, index
│   │       ├── commodity_intelligence.py # Price charts, correlations, FFN stats
│   │       ├── financial_pnl.py          # P&L waterfall, segment analysis
│   │       ├── scenario_simulation.py    # Monte Carlo, what-if builder
│   │       ├── market_monitor.py         # Live market/FX/crypto tracker
│   │       └── data_explorer.py          # Dataset catalog & quality report
│   │
│   ├── data/                      # Data layer
│   │   ├── data_loader.py         # Legacy pandas loader (backward compat)
│   │   ├── feature_engineering.py # Lag features, rolling stats, calendar encoding
│   │   ├── polars_pipeline.py     # Polars-native Parquet pipeline (primary)
│   │   ├── synthetic_generator.py # O-U process synthetic data generation
│   │   └── connectors/
│   │       ├── yfinance_connector.py  # Yahoo Finance — commodities, indices, FX
│   │       ├── ccxt_connector.py      # CCXT — Binance crypto exchange data
│   │       ├── fred_connector.py      # FRED — macroeconomic indicators
│   │       ├── commodity_api.py       # LME/Quandl (placeholder)
│   │       ├── erp_connector.py       # SAP S/4HANA (placeholder)
│   │       └── data_lake.py           # Snowflake/Databricks (placeholder)
│   │
│   ├── drivers/                   # Deterministic financial model
│   │   ├── revenue_drivers.py     # Revenue = Volume × Net Price
│   │   ├── cost_drivers.py        # COGS = f(BOM, Commodity Index)
│   │   ├── capital_drivers.py     # Depreciation & CapEx scheduling
│   │   └── financial_model.py     # Full P&L construction engine
│   │
│   ├── governance/                # Model governance & compliance
│   │   ├── audit_trail.py         # JSONL append-only audit logging
│   │   ├── bias_tracking.py       # Forecast bias detection & trending
│   │   └── explainability.py      # Natural-language forecast explanations
│   │
│   ├── models/                    # ML model implementations
│   │   ├── commodity_forecast.py  # 4-method commodity forecasting orchestrator
│   │   ├── commodity_forecast_xgboost.py  # XGBoost commodity model
│   │   ├── futures_curve.py       # Futures curve extraction (Method 3)
│   │   ├── commodity_scenarios.py # Scenario analysis & variance tracking (Method 4)
│   │   ├── demand_forecast.py     # XGBoost demand by segment
│   │   ├── price_elasticity.py    # Log-log Ridge regression
│   │   ├── inventory_risk.py      # Days-of-supply & stockout probability
│   │   └── model_registry.py      # Model versioning & persistence
│   │
│   └── simulation/                # Probabilistic simulation
│       ├── monte_carlo.py         # MC engine with VaR/CVaR
│       └── scenario_engine.py     # 7 preset + custom scenarios
│
├── tests/
│   ├── conftest.py                # Shared fixtures (session-scoped)
│   ├── test_commodity_forecast.py # 6 tests — SARIMAX, XGBoost, CV
│   ├── test_data_pipeline.py      # 12 tests — Polars, connectors, analytics
│   ├── test_financial_model.py    # 7 tests — revenue, COGS, P&L
│   ├── test_governance.py         # 4 tests — audit, bias tracking
│   └── test_simulation.py         # 5 tests — Monte Carlo, scenarios
│
├── .env.example                   # Environment variable template
├── .gitignore
├── pyproject.toml                 # Project metadata & dependencies
└── README.md
```

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Data Processing** | Polars + Parquet | High-performance DataFrame ops, columnar storage |
| **ML / Forecasting** | SARIMAX, XGBoost, scikit-learn | Time-series, gradient boosted, futures curve & scenario forecasting |
| **Real-Time Data** | yfinance | Commodity futures, indices, FX — free, no API key |
| **Crypto Data** | ccxt (Binance) | Real-time crypto exchange data (BTC, ETH, SOL...) |
| **Macro Data** | fredapi | Federal Reserve economic indicators (free API key) |
| **Financial Analytics** | FFN | CAGR, Sharpe, Sortino, drawdown analysis |
| **API** | FastAPI + Uvicorn | Async REST API with Swagger/OpenAPI docs |
| **Dashboard** | Streamlit | 6-page interactive executive dashboard |
| **Visualization** | Plotly | Interactive charts with dark theme |
| **Config** | PyYAML + python-dotenv | YAML config + environment variables |
| **Testing** | pytest | 34 tests across 5 test files |
| **Logging** | Loguru | Structured logging with rotation |
| **Serialization** | Parquet (zstd) | Compressed columnar storage |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd gic-plan-to-perform

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your FRED API key (optional but recommended):
# FRED_API_KEY=your_key_here  (free at https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 3. Generate Synthetic Data (Instant — No API keys needed)

```bash
python scripts/generate_data.py
```

### 4. Fetch Real-World Data (Recommended)

```bash
python scripts/fetch_data.py
```

This pulls live data from:
- **Yahoo Finance**: 8 commodity proxies, 6 market indices, 4 FX pairs
- **CCXT/Binance**: 6 crypto assets (BTC, ETH, SOL, XRP, BNB, AVAX)
- **FRED**: 11 macro indicators (requires API key in `.env`)

### 5. Train Models

```bash
python scripts/train_models.py
```

### 6. Run Full Pipeline

```bash
python scripts/run_pipeline.py
```

### 6b. Run Commodity Forecast Pipeline (12 Materials, 4 Methods)

```bash
python scripts/run_commodity_pipeline.py
```

This runs the full 8-stage commodity pipeline:
1. Data generation (12 commodities + macro indicators)
2. Model training (SARIMAX + XGBoost per commodity with cross-validation)
3. Forecast generation (all 4 methods: SARIMAX, XGBoost, Futures Curve, Scenario)
4. Multi-method comparison table
5. BOM-weighted commodity index
6. Variance tracking & monthly update (with >5% alert / >10% escalation)
7. Macro scenario stress tests (Bear / Base / Bull)
8. Governance & audit trail

### 7. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

### 8. Launch API Server

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
# Swagger docs at http://127.0.0.1:8000/docs
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Summary** | CFO-level KPIs: Revenue, Margin, COGS, Net Income. Market risk assessment, active alerts, commodity index trend |
| **Commodity Intelligence** | Individual price charts with MA/Bollinger bands. Correlation heatmap. Performance analytics (CAGR, Sharpe, drawdown). BOM-weighted commodity index |
| **Financial P&L** | P&L waterfall chart. Segment revenue/volume breakdown. Monthly trend analysis. Commodity cost sensitivity |
| **Scenario Simulation** | Interactive Monte Carlo (1K–50K sims). Custom what-if builder with 6 adjustable drivers. Preset scenario comparison |
| **Market Monitor** | Live market indices, FX rates, crypto prices. FRED macro indicators. Real-time data from 3 sources |
| **Data Explorer** | Dataset catalog (external, parquet, synthetic). Data preview with statistics. Quality report (nulls, date coverage) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/models` | List loaded models |
| `POST` | `/forecast/commodity` | Commodity price forecast (SARIMAX) |
| `POST` | `/forecast/commodity-index` | BOM-weighted index forecast |
| `POST` | `/forecast/elasticity` | Price elasticity estimation |
| `POST` | `/simulation/scenario` | Run Monte Carlo for a scenario |
| `GET` | `/simulation/presets` | List preset scenarios |
| `POST` | `/simulation/compare-presets` | Compare all preset scenarios |

### Example: Commodity Forecast

```bash
curl -X POST http://localhost:8000/forecast/commodity \
  -H "Content-Type: application/json" \
  -d '{"commodity": "Lithium", "horizon_months": 12}'
```

### Example: What-If Scenario

```bash
curl -X POST http://localhost:8000/simulation/scenario \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Lithium Crisis",
    "demand_shock": -0.05,
    "commodity_shock": 0.40,
    "n_simulations": 10000
  }'
```

---

## Data Sources

### Real-Time (Free, No API Key)

| Source | Data | Refresh |
|--------|------|---------|
| **Yahoo Finance** | Commodity ETFs (LIT, REMX, SLX, HG=F, PPLT), Market indices (S&P 500, VIX, DJI), FX (EUR, GBP, JPY, CNY), Crude oil, Gold | On-demand |
| **CCXT / Binance** | BTC, ETH, SOL, XRP, BNB, AVAX — OHLCV candles | On-demand |

### Free with API Key

| Source | Data | How to Get |
|--------|------|------------|
| **FRED** | Fed Funds Rate, Treasury yields, CPI, PPI, GDP, Unemployment, Consumer Sentiment, WTI Oil, Gold, FX | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |

### Placeholder (Enterprise)

| Source | Status |
|--------|--------|
| SAP S/4HANA | Connector stub ready |
| Snowflake / Databricks | Connector stub ready |
| Anaplan | Connector stub ready |
| Bloomberg Terminal | Connector stub ready |

---

## Financial Model

### P&L Construction

```
Revenue     = Σ(Volume_segment × Net_Price × (1 - Incentive%))
COGS        = Revenue × Base_COGS_% × (1 + Commodity_Impact × Material_Fraction)
Gross Margin = Revenue - COGS
Warranty     = Revenue × 2.5%
Depreciation = CapEx / Useful_Life / 12
Op. Income   = Gross Margin - Warranty - Depreciation
Tax          = max(0, Op. Income × 21%)
Net Income   = Op. Income - Tax
```

### Commodity Index

BOM-weighted composite index normalized to base 100:

```
Index(t) = Σ(w_i × Price_i(t) / Price_i(0) × 100) / Σ(w_i)
```

Where weights reflect Bill of Materials cost allocation:
- Steel: 22% | Lithium: 18% | Aluminum: 12% | Cobalt: 7%
- Copper: 6% | Nickel: 5% | Platinum: 4% | Natural Gas: 4%
- Palladium: 3% | Polypropylene: 3% | Rhodium: 2% | ABS Resin: 2%

### Monte Carlo Simulation

- **10,000 simulations** per scenario (configurable up to 50K)
- **Demand shocks**: Normal distribution (σ = 10%)
- **Commodity shocks**: Student's t-distribution (df=5, σ = 20%) — captures fat tails
- **Risk metrics**: VaR and CVaR at 95% confidence
- **7 preset scenarios**: Base, Bull, Bear, Commodity Crisis, Lithium+15%, EU Demand-8%, Stagflation

---

## Commodities Tracked (12 JLR-Relevant Materials)

| Commodity | Category | BOM Weight | Preferred Method | Primary Driver | Risk Flag |
|-----------|----------|------------|-----------------|----------------|----------|
| Steel | Raw Material | 22% | ARIMA | Iron ore prices | Tariff exposure |
| Lithium | Battery Material | 18% | XGBoost | EV demand growth | Supply concentration |
| Aluminum | Raw Material | 12% | ARIMA | Energy cost | Carbon border tax |
| Cobalt | Battery Material | 7% | XGBoost | DRC supply disruptions | Geopolitical risk |
| Copper | Raw Material | 6% | ARIMA | Construction demand | Green transition demand |
| Nickel | Battery Material | 5% | XGBoost | Indonesia export policy | EV demand spike |
| Platinum | Precious Metal | 4% | XGBoost | Autocatalyst demand | Hydrogen economy |
| Natural Gas | Energy | 4% | ARIMA | TTF/Henry Hub spread | Geopolitical supply risk |
| Palladium | Precious Metal | 3% | XGBoost | Autocatalyst demand | Russian supply risk |
| Polypropylene | Polymer | 3% | ARIMA | Naphtha cost | Petrochemical cycle |
| Rhodium | Precious Metal | 2% | XGBoost | Emissions regulation | Extreme illiquidity |
| ABS Resin | Polymer | 2% | ARIMA | Styrene/butadiene prices | Petrochemical cycle |

---

## Model Details

### Commodity Forecast Model (4 Methods)

**Method 1 — SARIMAX (Baseline)**
- Order: (1,1,1)(1,1,1,12) — captures trend, seasonality, stationarity
- Best for: Stable commodities with seasonal patterns (Steel, Aluminum, Copper, Natural Gas, Polypropylene, ABS Resin)

**Method 2 — XGBoost (Macro-Driven)**
- 300 trees, depth 6, with engineered features:
  - Lag features: 1, 3, 6, 12 months
  - Rolling statistics: 3, 6, 12 month MA and std
  - Percentage changes: 1, 3, 6 month momentum
  - Macro indicators: Manufacturing PMI, DXY, China PPI, Baltic Dry, US PPI, EV sales growth (with lags)
  - Calendar: cyclical month encoding
- Best for: Supply-shock-sensitive commodities (Platinum, Palladium, Rhodium, Lithium, Cobalt, Nickel)

**Method 3 — Futures Curve Extraction (Market-Implied)**
- Extracts forward prices from exchange-traded futures term structure
- Supports 7 liquid commodities: Steel, Aluminum, Copper, Platinum, Palladium, Nickel, Natural Gas
- Generates synthetic curves (contango for industrial metals, backwardation for PGMs, seasonal for energy)
- Up to 18-month forward price extraction

**Method 4 — Scenario Analysis (Expert + Model Hybrid)**
- Bear / Base / Bull price targets for each commodity (12-month horizon)
- Macro-driven probability weight shifting based on:
  - PMI (below 50 → overweight Bear, above 55 → overweight Bull)
  - DXY (strong dollar → hawkish on commodities)
  - Supply disruption risk (increases Bull probability)
- Weighted expected price = P(Bear) × Bear + P(Base) × Base + P(Bull) × Bull

### Variance Tracking & Monthly Update Process
- Monthly variance: `|Actual - Prior Forecast| / Prior Forecast`
- **> 5% variance**: Alert triggered
- **> 10% variance**: L6 Governance review escalation
- JSONL-based variance history with full audit trail
- 7-step monthly update workflow: Receive prices → Validate → Reforecast → Revise scenarios → Track variance → Escalate if needed → Publish

### Preset Scenarios
| Scenario | Demand | Commodity | FX |
|----------|--------|-----------|----|
| Base Case | 0% | 0% | 0% |
| Lithium +15% | 0% | +15% | 0% |
| EU Demand -8% | -8% | 0% | 0% |
| Commodity Crisis | -5% | +40% | +10% |
| Bull Market | +10% | -5% | -2% |
| Rate Cuts | +5% | -2% | 0% |
| Stagflation | -12% | +25% | +8% |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_commodity_forecast.py -v
```

34 tests across 5 test files covering:
- Commodity forecasting (SARIMAX, XGBoost, cross-validation)
- Financial model (revenue, COGS, P&L, scenarios)
- Monte Carlo simulation (distributions, VaR, scenario comparison)
- Governance (audit trail, bias tracking)
- Data pipeline (Polars, connectors, FFN analytics, market intelligence)

---

## Vehicle Segments (JLR Context)

| Segment | Models | Avg Price | Annual Volume |
|---------|--------|-----------|---------------|
| Luxury SUV | Range Rover, RR Sport | $105,000 | 80,000 |
| Premium SUV | Defender, Discovery | $72,000 | 120,000 |
| Performance | F-PACE, E-PACE | $58,000 | 65,000 |
| EV | I-PACE, Future EV | $82,000 | 45,000 |

---

## Configuration

All configuration is centralized in [`config/settings.yaml`](config/settings.yaml):

- **Commodities**: Names, units, categories, BOM weights, tickers
- **Vehicle Segments**: Models, pricing, volumes
- **Forecast**: Horizon, confidence levels, lag features
- **Financial**: COGS %, tax rate, warranty reserve, depreciation
- **Simulation**: Number of sims, seed, 7 preset scenarios
- **Governance**: Bias thresholds, audit retention, max override
- **Data Sources**: yfinance/ccxt/FRED enable flags, default periods
- **Dashboard**: Port, theme, refresh interval, page list

---

## Architecture Improvements Over Base Design

1. **Real-world data integration**: yfinance, CCXT, FRED replace synthetic-only pipeline
2. **Polars + Parquet**: 10x faster data processing vs pandas/CSV with lazy evaluation
3. **Fat-tailed distributions**: Monte Carlo uses Student's t (df=5) for commodity shocks
4. **Ornstein-Uhlenbeck synthetic data**: Mean-reverting process for realistic fallback data
5. **Cross-validation**: Expanding-window time-series CV prevents look-ahead bias
6. **BOM-weighted Commodity Index**: Material cost weights from actual bill-of-materials
7. **FFN performance analytics**: Industry-standard financial metrics (CAGR, Sharpe, Sortino)
8. **Market intelligence engine**: Macro regime detection, alert generation, risk scoring
9. **6-page Streamlit dashboard**: CFO-grade interactive visualization
10. **Multi-source data pipeline**: Parquet → external → raw → synthetic fallback chain
11. **Append-only audit trail**: JSONL format for immutable governance logging
12. **Model registry with versioning**: Save/load/compare model versions over time

---

## Roadmap

- [ ] Real-time streaming with WebSocket price feeds
- [ ] Prophet / LSTM models for improved long-horizon forecasting
- [ ] Bloomberg Terminal integration (enterprise connector)
- [ ] Anaplan bi-directional sync for financial planning
- [ ] Docker + Kubernetes deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Role-based access control (RBAC)
- [ ] Automated retraining scheduler
- [ ] PDF report generation for board presentations
- [ ] Multi-currency P&L with hedging strategy optimizer

---

## License

Internal use — proprietary.
