"""
Market Data Provider — single source of truth for commodity & macro prices.

All components (realtime feed, forecast models, API routes) call this module.
Data source priority (evaluated on each hourly cache refresh):
  1. In-memory cache           (< 1 hour old — avoids repeated yfinance calls)
  2. yfinance live pull        (when market_data_source=live in settings.yaml)
     └─ saves result to data/raw/commodity_prices.csv as a side-effect cache
  3. data/raw/commodity_prices.csv   (stale CSV from previous fetch or fetch_data.py)
  4. data/synthetic/commodity_prices.csv  (last-resort generated fallback)

Changing config/settings.yaml → market_data_source propagates everywhere because
all consumers call get_commodity_prices() / get_latest_prices().
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

# Tracks which source was actually used in the last _load_prices() call.
_data_source_used: str = "Unknown"


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
    """Return the most recent month's prices as {commodity_name: price}."""
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
    """Human-readable label for the active data source."""
    return _data_source_used


def invalidate_cache() -> None:
    """Force the next call to re-fetch from yfinance / disk."""
    global _cache, _cache_ts
    with _lock:
        _cache = None
        _cache_ts = 0.0


# ── Internal loader ────────────────────────────────────────────────────────────


def _load_prices() -> pd.DataFrame:
    """Load commodity prices according to the priority chain.

    Always tries yfinance live first (when market_data_source=live) so the
    dashboard reflects current prices without needing to run fetch_data.py.
    Saves the fresh data to data/raw/ asynchronously as a cache.
    """
    global _data_source_used

    root = get_project_root()
    raw_path = root / "data" / "raw" / "commodity_prices.csv"
    synth_path = root / "data" / "synthetic" / "commodity_prices.csv"
    settings = get_settings()

    # 1. Live pull from yfinance (priority — direct, no manual script required)
    if settings.get("market_data_source", "live") == "live":
        try:
            from src.data.connectors.yfinance_connector import fetch_commodity_prices
            df_pl = fetch_commodity_prices()
            if not df_pl.is_empty():
                df = df_pl.to_pandas()
                df_sorted = df.sort_values("date").reset_index(drop=True)
                logger.info("MarketDataProvider: live pull from Yahoo Finance succeeded")
                _data_source_used = "Live (Yahoo Finance)"
                _save_to_raw_async(df_sorted, raw_path)
                return df_sorted
        except Exception as exc:
            logger.warning(f"MarketDataProvider: yfinance live pull failed ({exc}); trying CSV")

    # 2. Cached CSV (written by previous yfinance pull or scripts/fetch_data.py)
    if raw_path.exists():
        logger.info("MarketDataProvider: reading data/raw/commodity_prices.csv (cached)")
        df = pd.read_csv(raw_path, parse_dates=["date"])
        _data_source_used = "Cached (Yahoo Finance)"
        return df.sort_values("date").reset_index(drop=True)

    # 3. Synthetic fallback
    if synth_path.exists():
        logger.warning("MarketDataProvider: using synthetic commodity prices (no real data available)")
        df = pd.read_csv(synth_path, parse_dates=["date"])
        _data_source_used = "Synthetic"
        return df.sort_values("date").reset_index(drop=True)

    raise FileNotFoundError(
        "No commodity price data found. "
        "Set market_data_source: live in config/settings.yaml for automatic live fetch, "
        "or run `python scripts/generate_data.py` for synthetic fallback."
    )


def _save_to_raw_async(df: pd.DataFrame, raw_path: Path) -> None:
    """Persist fresh yfinance data to data/raw/ in a background thread.

    Non-blocking — the caller gets the data immediately. The CSV write
    ensures subsequent requests (e.g., after server restart) use the cached
    real data even if yfinance is temporarily unavailable.
    """
    def _write() -> None:
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path, index=False)
            logger.info(f"MarketDataProvider: saved {len(df)} rows to {raw_path.name}")
        except Exception as exc:
            logger.warning(f"MarketDataProvider: could not write {raw_path} ({exc})")

    threading.Thread(target=_write, daemon=True).start()
