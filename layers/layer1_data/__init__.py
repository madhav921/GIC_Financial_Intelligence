"""
Layer 1 — Data Architecture

Responsibilities:
  - Fetch real-world commodity prices (Yahoo Finance), macro data (FRED), crypto (CCXT/Binance)
  - Generate synthetic Ornstein-Uhlenbeck fallback data for illiquid commodities
  - Build high-performance Polars/Parquet pipeline (10x faster than pandas/CSV)
  - Route data sources: Parquet → External → Raw CSV → Synthetic (priority chain)
  - Engineer ML features: lags, rolling stats, momentum, RSI, MACD, calendar encoding

Key modules (in src/data/):
  connectors/yfinance_connector.py  — Yahoo Finance (9 commodities, indices, FX)
  connectors/fred_connector.py      — FRED macro indicators
  connectors/ccxt_connector.py      — Binance crypto OHLCV
  polars_pipeline.py                — Parquet columnar storage with lazy evaluation
  synthetic_generator.py            — O-U mean-reverting synthetic prices
  feature_engineering.py            — Lag, rolling, momentum, macro features
  data_router.py                    — Source-agnostic data routing protocol
"""
from layers.layer1_data.controller import DataLayerController
__all__ = ["DataLayerController"]
