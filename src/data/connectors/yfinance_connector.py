"""
Yahoo Finance connector — real-time & historical commodity, equity, and FX data.

Commodities tracked via futures/ETF tickers:
  - GC=F  (Gold), SI=F (Silver), CL=F (Crude Oil), HG=F (Copper)
  - ALI=F (Aluminum futures proxy), ^GSPC (S&P 500)
  - LIT (Lithium ETF), COPX (Copper Miners ETF)
  - PALL (Palladium ETF), PPLT (Platinum ETF)
  - DBA (Agriculture), REMX (Rare Earth Minerals)
  - USDJPY=X, EURUSD=X, GBPUSD=X (FX pairs)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import yfinance as yf
from loguru import logger

# ── Commodity & Market Tickers ─────────────────────────────────────
COMMODITY_TICKERS = {
    "Lithium":   "LIT",        # Global X Lithium & Battery Tech ETF
    "Cobalt":    "REMX",       # VanEck Rare Earth/Strategic Metals ETF (proxy)
    "Nickel":    "JJN",        # iPath Nickel ETN (proxy via broader basket)
    "Aluminum":  "JJU",        # iPath Aluminum ETN / ALI=F proxy
    "Steel":     "SLX",        # VanEck Steel ETF
    "Copper":    "HG=F",       # Copper Futures (COMEX)
    "Platinum":  "PPLT",       # abrdn Platinum Shares ETF
    "Rubber":    "RUBB.L",     # Rubber futures proxy
}

# Broader market context for macro signals
MARKET_TICKERS = {
    "SP500":     "^GSPC",
    "DowJones":  "^DJI",
    "VIX":       "^VIX",
    "OilCrude":  "CL=F",
    "Gold":      "GC=F",
    "UST10Y":    "^TNX",       # 10-Year US Treasury Yield
}

FX_TICKERS = {
    "USD_GBP": "GBPUSD=X",
    "USD_EUR": "EURUSD=X",
    "USD_JPY": "USDJPY=X",
    "USD_CNY": "USDCNY=X",
}

CRYPTO_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def fetch_commodity_prices(
    period: str = "7y",
    interval: str = "1mo",
) -> pl.DataFrame:
    """
    Fetch historical commodity prices from Yahoo Finance.

    Args:
        period: yfinance period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data granularity (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

    Returns:
        Polars DataFrame with date + commodity columns
    """
    logger.info(f"Fetching commodity prices from Yahoo Finance (period={period}, interval={interval})")
    all_tickers = list(COMMODITY_TICKERS.values())
    ticker_to_name = {v: k for k, v in COMMODITY_TICKERS.items()}

    try:
        data = yf.download(all_tickers, period=period, interval=interval, progress=False)
        if data.empty:
            logger.warning("No data returned from Yahoo Finance")
            return pl.DataFrame()

        # Extract adjusted close prices
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            close_data = data["Close"]
        else:
            close_data = data[["Close"]]

        # Convert to Polars
        pdf = close_data.reset_index()
        pdf.columns = ["date"] + [ticker_to_name.get(c, c) for c in close_data.columns]
        df = pl.from_pandas(pdf)

        # Clean: forward-fill and drop nulls at start
        commodity_cols = [c for c in df.columns if c != "date"]
        df = df.with_columns([
            pl.col(c).forward_fill().alias(c) for c in commodity_cols
        ]).drop_nulls(subset=commodity_cols[:1])

        logger.info(f"Fetched {df.height} rows × {len(commodity_cols)} commodities from Yahoo Finance")
        return df

    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed: {e}")
        return pl.DataFrame()


def fetch_market_data(
    period: str = "7y",
    interval: str = "1mo",
) -> pl.DataFrame:
    """Fetch broader market indices (S&P 500, VIX, Oil, Gold, Treasury yields)."""
    logger.info("Fetching market index data from Yahoo Finance")
    all_tickers = list(MARKET_TICKERS.values())
    ticker_to_name = {v: k for k, v in MARKET_TICKERS.items()}

    try:
        data = yf.download(all_tickers, period=period, interval=interval, progress=False)
        if data.empty:
            return pl.DataFrame()

        if isinstance(data.columns, __import__("pandas").MultiIndex):
            close_data = data["Close"]
        else:
            close_data = data[["Close"]]

        pdf = close_data.reset_index()
        pdf.columns = ["date"] + [ticker_to_name.get(c, c) for c in close_data.columns]
        df = pl.from_pandas(pdf)

        cols = [c for c in df.columns if c != "date"]
        df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in cols])
        logger.info(f"Fetched {df.height} rows of market data")
        return df

    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")
        return pl.DataFrame()


def fetch_fx_rates(
    period: str = "7y",
    interval: str = "1mo",
) -> pl.DataFrame:
    """Fetch historical FX rates."""
    logger.info("Fetching FX rates from Yahoo Finance")
    all_tickers = list(FX_TICKERS.values())
    ticker_to_name = {v: k for k, v in FX_TICKERS.items()}

    try:
        data = yf.download(all_tickers, period=period, interval=interval, progress=False)
        if data.empty:
            return pl.DataFrame()

        if isinstance(data.columns, __import__("pandas").MultiIndex):
            close_data = data["Close"]
        else:
            close_data = data[["Close"]]

        pdf = close_data.reset_index()
        pdf.columns = ["date"] + [ticker_to_name.get(c, c) for c in close_data.columns]
        df = pl.from_pandas(pdf)

        cols = [c for c in df.columns if c != "date"]
        df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in cols])
        logger.info(f"Fetched {df.height} rows of FX data")
        return df

    except Exception as e:
        logger.error(f"FX data fetch failed: {e}")
        return pl.DataFrame()


def fetch_single_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> pl.DataFrame:
    """Fetch a single ticker's OHLCV data."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return pl.DataFrame()
        pdf = hist.reset_index()
        return pl.from_pandas(pdf)
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pl.DataFrame()


def fetch_all_market_data(period: str = "7y", interval: str = "1mo") -> dict[str, pl.DataFrame]:
    """Fetch all market data categories in one call."""
    return {
        "commodities": fetch_commodity_prices(period=period, interval=interval),
        "market_indices": fetch_market_data(period=period, interval=interval),
        "fx_rates": fetch_fx_rates(period=period, interval=interval),
    }
