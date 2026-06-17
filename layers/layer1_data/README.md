# Layer 1 — Data Architecture

**Controller:** `DataLayerController` (`controller.py`)
**Entry point:** `controller.load_all()` → `(commodity_df, macro_df, sales_df, bom_df)`

## What this layer does

Handles all data acquisition, transformation, and routing for the GIC platform.
It provides a single, source-agnostic DataFrame interface regardless of whether
data comes from live APIs, cached Parquet files, CSV snapshots, or synthetic generation.

## Data source priority chain

```
Parquet (fastest) → Real CSV → Synthetic O-U fallback
```

## Key src/ modules

| Module | Role |
|---|---|
| `src/data/connectors/yfinance_connector.py` | Yahoo Finance: 9 commodities, equity indices, FX |
| `src/data/connectors/fred_connector.py` | FRED: PMI, CPI, Fed Funds, yield curve |
| `src/data/connectors/ccxt_connector.py` | Binance: crypto OHLCV via CCXT |
| `src/data/polars_pipeline.py` | Parquet columnar storage, lazy evaluation |
| `src/data/synthetic_generator.py` | Ornstein-Uhlenbeck mean-reverting synthetic prices |
| `src/data/feature_engineering.py` | Lags, rolling stats, RSI, MACD, calendar encoding |
| `src/data/data_router.py` | Source-agnostic routing protocol |
