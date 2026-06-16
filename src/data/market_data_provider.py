"""
Market Data Provider — single source of truth for commodity & macro prices.

All components (realtime feed, forecast models, API routes) call this module.
The data source priority is:
  1. In-memory cache (< 1 hour old, avoids repeated yfinance calls)
  2. data/raw/commodity_prices.csv   (populated by scripts/fetch_data.py)
  3. YFinanceMarketSource live pull  (when market_data_source=live in settings)
  4. data/synthetic/commodity_prices.csv  (last-resort fallback only)

Changing config/settings.yaml → market_data_source propagates everywhere
because all consumers go through get_commodity_prices() / get_latest_prices().
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_project_root, get_settings

_lock = threading.Lock()
_cache: dict[str, object] | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 3600  # seconds — re-fetch live data at most once per hour


# ── Public API ─────────────────────────────────────────────────────────────────


def get_commodity_prices(force_refresh: bool = False) -> pd.DataFrame:
    """Return full monthly commodity price history as a DataFrame.

    Columns: date + commodity names (Steel, Aluminum, Copper, Lithium, ...).
    Prices are in the units defined by COMMODITY_TICKERS in yfinance_connector.py.
    """
    global _cache, _cache_ts

    with _lock:
        age = time.time() - _cache_ts
        if not force_refresh and _cache is not None and age < _CACHE_TTL:
            return _cache["commodity_prices"]  # type: ignore[return-value]

        df = _load_prices()
        _cache = {"commodity_prices": df}
        _cache_ts = time.time()
        logger.info(f"MarketDataProvider: cache refreshed ({len(df)} rows, latest {df['date'].max()})")
        return df


def get_latest_prices() -> dict[str, float]:
    """Return the most recent month's prices as {commodity_name: price}.

    Used by the realtime feed for seeding, and by dashboard snapshot endpoints.
    """
    df = get_commodity_prices()
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        col: float(last[col])
        for col in df.columns
        if col != "date" and pd.notna(last[col]) and float(last[col]) > 0
    }


def get_data_source_label() -> str:
    """Human-readable label for what source is active ('Real (Yahoo Finance)' or 'Synthetic')."""
    root = get_project_root()
    raw_path = root / "data" / "raw" / "commodity_prices.csv"
    if raw_path.exists():
        return "Real (Yahoo Finance)"
    settings = get_settings()
    source = settings.get("market_data_source", "synthetic")
    return "Live (Yahoo Finance)" if source == "live" else "Synthetic"


def invalidate_cache() -> None:
    """Force the next call to re-fetch from disk / yfinance."""
    global _cache, _cache_ts
    with _lock:
        _cache = None
        _cache_ts = 0.0


# ── Internal loader ────────────────────────────────────────────────────────────


def _load_prices() -> pd.DataFrame:
    root = get_project_root()
    raw_path = root / "data" / "raw" / "commodity_prices.csv"
    synth_path = root / "data" / "synthetic" / "commodity_prices.csv"

    # 1. Real data from fetch_data.py
    if raw_path.exists():
        logger.info("MarketDataProvider: reading data/raw/commodity_prices.csv (Yahoo Finance)")
        df = pd.read_csv(raw_path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    # 2. Live pull from yfinance (only when market_data_source=live)
    settings = get_settings()
    if settings.get("market_data_source", "live") == "live":
        try:
            from src.data.connectors.yfinance_connector import fetch_commodity_prices
            df_pl = fetch_commodity_prices()
            if not df_pl.is_empty():
                df = df_pl.to_pandas()
                logger.info("MarketDataProvider: live pull from Yahoo Finance succeeded")
                return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"MarketDataProvider: yfinance live pull failed ({exc})")

    # 3. Synthetic fallback
    if synth_path.exists():
        logger.warning("MarketDataProvider: using synthetic commodity prices (no real data found)")
        df = pd.read_csv(synth_path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    raise FileNotFoundError(
        "No commodity price data found. "
        "Run `python scripts/fetch_data.py` for real data or "
        "`python scripts/generate_data.py` for synthetic fallback."
    )
