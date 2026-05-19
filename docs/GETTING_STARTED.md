# Getting Started — GIC Financial Intelligence Platform

Welcome! This guide will get you up and running in **5 minutes**.

---

## What Is GIC?

**GIC** is an AI-powered financial intelligence engine that transforms commodity market data into real-time P&L impacts. It combines:

- **Real market data**: Yahoo Finance, FRED, CCXT, plus synthetic data generators
- **ML forecasting**: SARIMAX + XGBoost ensemble with regime detection
- **Financial modeling**: Deterministic COGS/Revenue drivers plus Monte Carlo risk simulation
- **Interactive dashboard**: Streamlit interface with live commodity shocks, hedging tools, and scenario analysis

**Built for**: CFOs, commodity managers, treasury teams  
**Core value**: Reduces EBIT uncertainty by £18.4M/yr through better forecasts + optimal hedging

---

## Installation (2 minutes)

### Prerequisites
- **Python 3.10+** installed
- **Git** for cloning the repository

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/madhav921/GIC_Financial_Intelligence.git
   cd GIC_Financial_Intelligence
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download real market data** (optional, ~30 seconds)
   ```bash
   python scripts/fetch_data.py
   ```
   This fetches 7 years of commodity prices from Yahoo Finance + macro indicators from FRED.

---

## Your First 2 Runs

### Run 1: Generate the Executive Intelligence Report

```bash
python scripts/generate_executive_report.py
```

**What happens:**
- Loads real commodity data (data/raw/commodity_prices.csv)
- Computes commodity statistics, correlations, and financial impacts
- Generates **docs/EXECUTIVE_INTELLIGENCE_REPORT.md** (766-line board-ready report)

**Output**: Opens automatically or view at:  
`docs/EXECUTIVE_INTELLIGENCE_REPORT.md`

**What you'll see:**
- BOM-weighted commodity index (currently +84.7% YoY)
- Forecast accuracy tables (real CV MAPE metrics)
- Scenario P&L impacts (Bear/Base/Bull cases)
- Macro-commodity correlations
- Risk metrics (VaR, Monte Carlo)

---

### Run 2: Start the Interactive Dashboard

```bash
$env:PYTHONIOENCODING="utf-8"
python -m streamlit run src/dashboard/app.py --server.port 8502
```

**What opens:**
- Local dashboard at `http://localhost:8502`
- 8-page interactive interface

**Try these (in order):**

1. **📊 Executive Summary** (default)
   - KPI cards (commodity index, COGS impact)
   - Real commodity prices + forecasts
   - Scenario waterfall

2. **📋 Intelligence Report**
   - Full downloadable board report
   - Commodity deep dives
   - Accuracy metrics per commodity

3. **🌍 Commodity Intelligence**
   - Individual commodity forecast with 95% confidence bands
   - Scenario + sensitivity sliders
   - Move the **Lithium** slider +20% → watch EBIT impact update in <1 second

4. **💰 Financial P&L**
   - Revenue, COGS, margin drivers
   - Monte Carlo simulation results (10,000 runs)
   - Risk breakdown (commodity vol, demand vol, FX)

---

## Understanding the Output

### Executive Intelligence Report (Key Sections)

| Section | What It Shows | How to Use |
|---------|---|---|
| **Section 1: Market Snapshot** | Current commodity prices + YoY changes | Understand current market state |
| **Section 2: Index & Trends** | BOM-weighted commodity index time series | Monitor portfolio inflation trend |
| **Section 3: Forecast Accuracy** | Real model CV MAPE + directional accuracy | Understand reliability by commodity |
| **Section 4: Macro Correlations** | Which economic indicators drive prices | Scenario building context |
| **Section 5: Scenario P&L** | COGS + EBIT impact under 3 scenarios | Budget planning, guidance ranges |
| **Section 6: Risk Metrics** | VaR, Monte Carlo distribution | CFO-grade risk quantification |

### Dashboard Pages Explained

- **Commodity Intelligence**: Real-time shock calculator — move a commodity slider and watch EBIT impact update
- **Financial P&L**: Distribution of possible outcomes (not just base case)
- **Market Monitor**: Correlation matrix + regime detection
- **Backtesting**: Historical forecast accuracy

---

## Key Files & Directories

```
GIC_Financial_Intelligence/
├── config/settings.yaml              ← Adjust BOM weights, scenario prices here
├── data/
│   ├── raw/                          ← Real commodity prices (Yahoo Finance)
│   │   └── commodity_prices.csv      ← 7-year historical, 12 commodities
│   ├── synthetic/                    ← Generated JLR-like data
│   │   └── sales_data.csv
│   └── external/                     ← Cached parquet (faster loads)
├── src/
│   ├── data/                         ← Polars pipeline, data connectors
│   ├── models/                       ← SARIMAX, XGBoost, regime detection
│   ├── drivers/                      ← Financial model (COGS, revenue)
│   ├── simulation/                   ← Monte Carlo engine
│   └── dashboard/                    ← Streamlit multi-page app
├── scripts/
│   ├── fetch_data.py                 ← Download real market data
│   ├── generate_executive_report.py  ← Build the board-ready report
│   ├── run_commodity_pipeline.py     ← Train all models (one-time, ~5 min)
│   └── run_full_architecture.py      ← End-to-end (data → models → P&L → MC)
├── docs/
│   ├── EXECUTIVE_INTELLIGENCE_REPORT.md  ← Generated output (read this!)
│   ├── GETTING_STARTED.md            ← This file
│   ├── ARCHITECTURE_GUIDE.md         ← Deep technical walkthrough
│   └── OUTPUT_GUIDE.md               ← Interpreting the numbers
└── tests/                            ← 34 unit tests
```

---

## Common Tasks

### Update commodity prices with latest market data
```bash
python scripts/fetch_data.py
```
This fetches the latest commodities, macro, and FX data from live sources.

### Retrain all models (takes ~5 minutes)
```bash
python scripts/run_commodity_pipeline.py
```
Trains SARIMAX+XGBoost models for all 12 commodities using real data.

### Run full architecture (data → models → P&L → Monte Carlo)
```bash
python scripts/run_full_architecture.py
```
End-to-end validation with accuracy backtesting.

### Run tests
```bash
pytest tests/ -v
```
Validates all 34 test cases (should pass in <10 sec).

---

## Troubleshooting

### Dashboard won't start
```bash
# Make sure port 8502 is free
netstat -ano | findstr :8502  # Windows
lsof -i :8502                 # macOS/Linux

# Try a different port
python -m streamlit run src/dashboard/app.py --server.port 8503
```

### "Module not found" error
```bash
# Ensure venv is activated, then reinstall
pip install -r requirements.txt --force-reinstall
```

### Models missing or old
```bash
# Retrain all models
python scripts/run_commodity_pipeline.py

# This saves to: models/saved/{commodity}_xgb_*/
```

### Data fetching fails
```bash
# If Yahoo Finance API is rate-limited, use cached data
ls data/raw/  # Check if commodity_prices.csv exists
```

---

## Next Steps

1. **Read the full report**: `docs/EXECUTIVE_INTELLIGENCE_REPORT.md`
2. **Explore the dashboard**: Pages 2–4 are most interactive
3. **Understand the architecture**: `docs/ARCHITECTURE_GUIDE.md`
4. **See the validation evidence**: `docs/FULL_ARCHITECTURE_RUN.md` — actual 2024 backtest numbers, P&L by segment, Monte Carlo calibration proof
5. **Try different scenarios**: Use `config/settings.yaml` to adjust BOM weights or base case prices
6. **Review the code**: `src/drivers/financial_model.py` shows exactly how P&L is calculated

---

## Questions?

| Topic | Document |
|-------|----------|
| Architecture & design decisions | [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) |
| Interpreting report numbers | [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) |
| Production readiness & roadmap | [../TECHNICAL_ASSESSMENT.md](../TECHNICAL_ASSESSMENT.md) |
| Actual 2024 backtest proof | [FULL_ARCHITECTURE_RUN.md](FULL_ARCHITECTURE_RUN.md) |
| Full project overview | [../README.md](../README.md) |
