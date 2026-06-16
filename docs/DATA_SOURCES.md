# Data Sources & Data Flow

> **GIC Financial Intelligence Platform — CFO-Level Intelligence**  
> Everything here describes **what data is used where, how it's fetched, and why each source is chosen.**

---

## Guiding Principle

> Real data first. Synthetic only as a last resort.

The platform is engineered to consume live market data by default. The fallback ladder is:

```
1. data/raw/         ← real data from fetch_data.py (Yahoo Finance, FRED, CCXT)
2. data/external/    ← same real data in Parquet format (faster reads)
3. data/synthetic/   ← generated via Ornstein-Uhlenbeck process (always available, never stale)
```

`src/data/data_loader.py → _resolve_path(name)` implements this: it tries `data/raw/{name}.csv` first, then `data/synthetic/{name}.csv`. Nothing in the application code ever hardcodes "use synthetic."

---

## Data Sources by Category

### 1. Commodity Prices
**Source:** Yahoo Finance (yfinance) via `src/data/connectors/yfinance_connector.py`  
**Script:** `python scripts/fetch_data.py`  
**Output:** `data/raw/commodity_prices.csv`  
**Coverage:** 9 of 12 commodities have exchange-traded proxies; 3 use synthetic O-U generation

| Commodity | Ticker | Source | Notes |
|-----------|--------|--------|-------|
| Steel | SLX | Yahoo Finance | ETF proxy for LME HRC |
| Aluminum | AA | Yahoo Finance | Alcoa proxy |
| Copper | HG=F | Yahoo Finance | LME 3M futures |
| Platinum | PL=F | Yahoo Finance | NYMEX futures |
| Palladium | PA=F | Yahoo Finance | NYMEX futures |
| Lithium | LIT | Yahoo Finance | Lithium ETF proxy |
| Cobalt | GLNCY | Yahoo Finance | Glencore proxy |
| Nickel | VALE | Yahoo Finance | Vale proxy |
| Natural Gas | NG=F | Yahoo Finance | Henry Hub futures |
| Rhodium | — | **Synthetic** | No exchange-traded instrument |
| Polypropylene | — | **Synthetic** | ICIS/Platts not freely available |
| ABS Resin | — | **Synthetic** | ICIS/Platts not freely available |

**Why Yahoo Finance?** Free, no API key required, 7-year monthly history, covers 9/12 commodities. Real exchange data (LME, CME, ICIS) requires paid subscriptions — Yahoo Finance proxies are well-correlated for planning purposes.

---

### 2. Macro Indicators
**Source:** FRED (fredapi) + Yahoo Finance macro proxies  
**Script:** `python scripts/fetch_data.py`  
**Output:** `data/raw/macro_indicators.csv`

| Indicator | FRED Series | Fallback | Used For |
|-----------|-------------|---------|----------|
| Fed Funds Rate | FEDFUNDS | Yahoo Finance | Discount rate, bond pricing |
| CPI Index | CPIAUCSL | Synthetic O-U | Inflation adjustment, real pricing |
| US PPI | PPIACO | Yahoo Finance | Input cost proxy |
| Industrial Production | INDPRO | Yahoo Finance | Manufacturing PMI proxy |
| USD/GBP | — | Yahoo Finance FX | Revenue conversion |
| USD/EUR | — | Yahoo Finance FX | European segment FX |
| DXY Index | — | Yahoo Finance | Dollar strength index |
| Oil Price (USD) | — | Yahoo Finance NG=F | Energy cost driver |

**Why FRED?** Free, authoritative US macro data with long history. Requires `FRED_API_KEY` env var; gracefully falls back to Yahoo Finance proxies if key not set.

---

### 3. FX Rates
**Source:** Yahoo Finance FX pairs  
**Script:** `python scripts/fetch_data.py`  
**Output:** `data/external/fx_rates.parquet`

| Pair | Why It Matters |
|------|----------------|
| USD/GBP | Main revenue currency conversion (JLR reports in GBP) |
| USD/EUR | European market exposure |
| USD/JPY | Japanese supplier invoicing |
| USD/CNY | Chinese manufacturing cost exposure |

**Real-time:** The WebSocket feed (`/ws/market`) provides tick-level FX updates seeded from these historical values with mean-reverting micro-walk.

---

### 4. Market Indices
**Source:** Yahoo Finance  
**Script:** `python scripts/fetch_data.py`  
**Output:** `data/external/market_indices.parquet`

| Index | Ticker | Use |
|-------|--------|-----|
| S&P 500 | ^GSPC | Macro regime indicator |
| VIX | ^VIX | Risk-off signal |
| Gold | GC=F | Safe-haven indicator |
| Crude Brent | BZ=F | Energy cost driver |
| EURO STOXX Auto | SX7P | Sector peer benchmark |

**Note:** These are displayed as reference data on the Market Monitor page (updated via `fetch_data.py`), not in the real-time feed.

---

### 5. Operational Data (Sales, BOM, Production)
**Current mode:** `operational_data_source: synthetic` (config/settings.yaml)  
**Synthetic generator:** `src/data/synthetic_generator.py`

To use **real** sales data:
1. Drop your file as `data/raw/sales_data.csv` (schema: `date, segment, volume, avg_price_usd`)
2. Change `config/settings.yaml` → `operational_data_source: parquet`
3. For SAP S/4HANA or Salesforce, set `operational_data_source: sap` or `salesforce` and implement `src/data/connectors/erp_connector.py`

JLR-calibrated synthetic volumes and prices are from public financials — they are realistic for demo and development.

---

### 6. Crypto / Alternative Data
**Source:** Binance via CCXT  
**Script:** `python scripts/fetch_data.py`  
**Output:** `data/external/crypto_prices.parquet`  
**Use:** Speculative correlation analysis, not in main P&L pipeline.

---

## Data Flow Through the Platform

```
scripts/fetch_data.py
│
├─ yfinance_connector.py   →  data/raw/commodity_prices.csv
│                          →  data/external/market_commodities.parquet
│                          →  data/external/market_indices.parquet
│                          →  data/external/fx_rates.parquet
│
├─ fred_connector.py       →  data/external/fred_macro.parquet
│                          (merged into data/raw/macro_indicators.csv)
│
└─ ccxt_connector.py       →  data/external/crypto_prices.parquet
         │
         ▼
src/data/data_loader.py        ← ALL pipeline components use this
    _resolve_path():
        1. Try data/raw/{name}.csv
        2. Fall back to data/synthetic/{name}.csv
         │
         ▼
         ├─ load_commodity_prices()  →  Layer 2 Intelligence (SARIMAX, SHAP)
         ├─ load_macro_indicators()  →  Layer 2 Intelligence (feature engineering)
         ├─ load_sales_data()        →  Layer 3 Financial (P&L waterfall)
         ├─ load_bom_data()          →  Layer 3 Financial (BOM-weighted COGS)
         └─ load_production_inventory() → Layer 3 Financial (capacity)
         │
         ▼
src/data/data_router.py        ← Market data (live prices at request time)
    market_data_source: live
        YFinanceMarketSource.get_commodity_prices()  →  real-time yfinance call
        YFinanceMarketSource.get_macro_indicators()  →  FRED or yfinance fallback
    market_data_source: parquet
        ParquetMarketSource  →  data/external/*.parquet  (fast, no network)
    market_data_source: synthetic
        _SyntheticMarketSource  →  generate_commodity_prices()
```

---

## API Endpoints & Their Data Sources

| Endpoint | Data Source | Real Data? |
|----------|-------------|-----------|
| `GET /pnl/annual` | `DataLayerController.load_all()` → DataLoader → `data/raw/` | ✅ when fetched |
| `POST /forecast/commodity` | `DataLoader.load_commodity_prices()` → SARIMAX+XGBoost | ✅ when fetched |
| `GET /forecast/commodity-index` | Same as above | ✅ when fetched |
| `GET /simulation/scenario` | `FinancialModel` + MC engine | ✅ (uses real commodity prices) |
| `GET /simulation/monthly-fan` | `MonteCarloEngine.run_monthly_fan()` | ✅ (MC driven by real vol) |
| `GET /simulation/variance-decomposition` | `MonteCarloEngine.decompose_variance()` | ✅ (MC driven by real vol) |
| `GET /insights/early-warning` | `EarlyWarningSystem` → DataLoader | ✅ when fetched |
| `GET /insights/variance-bridge` | `VarianceBridge` → DataLoader | ✅ when fetched |
| `GET /insights/feed` | `InsightEngine` → DataLoader | ✅ when fetched |
| `WS /ws/market` | `MarketFeed` seeded from `data/synthetic/commodity_prices.csv` | Simulated |
| `GET /realtime/snapshot` | `MarketFeed.snapshot()` | Simulated |
| `GET /pnl/regime` | `YFinanceMarketSource.get_commodity_prices()` live | ✅ live yfinance call |
| `GET /intelligence/change-points/{c}` | `DataLoader.load_commodity_prices()` | ✅ when fetched |
| `GET /intelligence/quantile-var` | `DataLoader.load_commodity_prices()` | ✅ when fetched |

---

## Frontend Data Source Indicators

Each page shows its live status:

| Page | Indicator | What It Means |
|------|-----------|---------------|
| Market Monitor | `● Connected` / `Simulated` | WebSocket to `/ws/market` live or client-side simulator |
| Commodity Intelligence | `● Live Forecast` | SARIMAX forecast from backend using `data/raw/` prices |
| Financial P&L | `● ` suffix on KPI titles | KPI strip from `/pnl/annual` using pipeline |
| Executive Summary | auto-refresh every 2s | Realtime context + insights feed |
| Scenario Simulation | MC Engine status tile | Backend Monte Carlo with real volatility |

When the backend is offline, all pages fall back to static/seeded mock data and show a yellow banner.

---

## How to Get Real Data Running

```bash
# 1. Install full dependencies (includes yfinance, fredapi, ccxt, polars)
pip install -e ".[full]"

# 2. (Optional) Set FRED API key for authoritative macro data
export FRED_API_KEY=your_key_here   # free at https://fred.stlouisfed.org/

# 3. Fetch real-world data (runs in ~2-3 minutes)
python scripts/fetch_data.py
# Writes: data/raw/commodity_prices.csv, data/raw/macro_indicators.csv
# Writes: data/external/*.parquet

# 4. Start the backend (now uses real data)
uvicorn src.api.app:app --reload --port 8000

# 5. (Optional) Train models on real data
python scripts/train_models.py
# Writes: models/saved/*.joblib

# 6. Start the frontend
cd frontend && npm start
```

After step 4, all API endpoints serve real commodity prices and real macro indicators. The forecast models train on 7 years of Yahoo Finance monthly data.

---

## Data Freshness & Update Cadence

| Dataset | Recommended Update | How |
|---------|-------------------|-----|
| Commodity prices | Weekly (Monday morning) | `python scripts/fetch_data.py` |
| Macro indicators | Monthly | `python scripts/fetch_data.py` |
| Forecast models | Monthly or on regime shift | `python scripts/train_models.py` |
| Realtime WebSocket | Continuous (2s ticks) | Automatic via `MarketFeed` |
| Audit trail | Append-only | Automatic via `AuditTrail` |

**Note:** The `fetch_data.py` script is idempotent — running it multiple times only overwrites with fresher data. Safe to run on a cron schedule.
