"""
FRED (Federal Reserve Economic Data) connector — free macroeconomic data.

Provides access to 800,000+ time series from the Federal Reserve Bank of St. Louis:
  - Interest rates (Federal Funds Rate, Treasury yields)
  - GDP growth, CPI/PPI inflation
  - Employment data
  - Consumer sentiment
  - Trade balances
  - Industrial production indices

Requires: Free API key from https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in .env
"""

from __future__ import annotations

import os

import polars as pl
from loguru import logger

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ── Key FRED Series for Financial Planning ──────────────────────
FRED_SERIES = {
    # Interest rates
    "FEDFUNDS":    "Fed Funds Rate",
    "DGS10":       "10Y Treasury Yield",
    "DGS2":        "2Y Treasury Yield",
    "T10Y2Y":      "10Y-2Y Spread (Yield Curve)",

    # Inflation
    "CPIAUCSL":    "CPI All Urban (SA)",
    "CPILFESL":    "Core CPI (ex Food/Energy)",
    "PPIACO":      "PPI All Commodities",

    # GDP & Output
    "GDP":         "Real GDP (Quarterly)",
    "INDPRO":      "Industrial Production Index",

    # Employment
    "UNRATE":      "Unemployment Rate",
    "PAYEMS":      "Total Nonfarm Payrolls",

    # Consumer
    "UMCSENT":     "University of Michigan Consumer Sentiment",
    "RSXFS":       "Retail Sales ex Food Services",

    # Commodity-related
    "DCOILWTICO":  "WTI Crude Oil Price",
    "GOLDAMGBD228NLBM": "Gold Price (London Fix)",

    # Currency
    "DEXUSEU":     "USD/EUR Exchange Rate",
    "DEXUSUK":     "USD/GBP Exchange Rate",
    "DEXCHUS":     "CNY/USD Exchange Rate",
}


def _get_fred() -> "Fred | None":
    """Initialize the FRED client with API key from environment."""
    if not FRED_AVAILABLE:
        logger.warning("fredapi not installed")
        return None

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key or api_key == "your_fred_api_key_here":
        logger.warning("FRED_API_KEY not set — FRED data unavailable. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    return Fred(api_key=api_key)


def fetch_fred_series(
    series_ids: list[str] | None = None,
    start_date: str = "2018-01-01",
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Fetch multiple FRED series and merge into a single DataFrame.

    Args:
        series_ids: List of FRED series IDs (defaults to all tracked series)
        start_date: Start date for data pull
        end_date: End date (defaults to today)

    Returns:
        Polars DataFrame with date + one column per series
    """
    fred = _get_fred()
    if fred is None:
        return pl.DataFrame()

    series_ids = series_ids or list(FRED_SERIES.keys())
    all_data = {}

    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            all_data[sid] = s
            logger.info(f"FRED: Fetched {sid} ({FRED_SERIES.get(sid, sid)}): {len(s)} observations")
        except Exception as e:
            logger.error(f"FRED: Failed to fetch {sid}: {e}")

    if not all_data:
        return pl.DataFrame()

    # Combine into a pandas DataFrame first, then convert
    import pandas as pd
    combined = pd.DataFrame(all_data)
    combined.index.name = "date"
    combined = combined.reset_index()

    df = pl.from_pandas(combined)
    # Forward-fill gaps (FRED series have different frequencies)
    value_cols = [c for c in df.columns if c != "date"]
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in value_cols])

    logger.info(f"FRED: Combined {len(value_cols)} series, {df.height} rows")
    return df


def fetch_macro_indicators(start_date: str = "2018-01-01") -> pl.DataFrame:
    """
    Fetch a curated set of macro indicators most relevant to commodity/financial planning.

    Returns monthly-frequency data covering interest rates, inflation, GDP, and sentiment.
    """
    key_series = [
        "FEDFUNDS", "DGS10", "T10Y2Y",
        "CPIAUCSL", "PPIACO",
        "INDPRO", "UNRATE", "UMCSENT",
        "DCOILWTICO",
        "DEXUSEU", "DEXUSUK",
    ]
    return fetch_fred_series(series_ids=key_series, start_date=start_date)


def fetch_commodity_ppi(start_date: str = "2018-01-01") -> pl.DataFrame:
    """Fetch Producer Price Index for commodities — leading indicator for COGS."""
    ppi_series = [
        "PPIACO",    # All Commodities
        "WPU10",     # Metals & Metal Products
        "WPU06",     # Chemicals & Allied Products
        "WPU01",     # Farm Products
    ]
    return fetch_fred_series(series_ids=ppi_series, start_date=start_date)
