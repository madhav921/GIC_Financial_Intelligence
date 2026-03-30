"""
Yahoo Finance connector — real-time & historical commodity, equity, and FX data.

Fetches real market data for JLR commodity forecasting:
  Direct futures:  HG=F (Copper), PL=F (Platinum), PA=F (Palladium), NG=F (Natural Gas)
  ETF proxies:     SLX (Steel), LIT (Lithium), AA (Aluminum), VALE (Nickel), GLNCY (Cobalt)
  Market indices:  ^GSPC, ^VIX, ^DJI, CL=F, GC=F, ^TNX
  Macro proxies:   DX-Y.NYB (DXY), CL=F (Oil), ^TNX (10Y yield)
  FX rates:        GBPUSD=X, EURUSD=X, USDJPY=X, USDCNY=X
"""

from __future__ import annotations

import numpy as np
import polars as pl
import yfinance as yf
from loguru import logger


# ── 12 JLR Commodities — best available yfinance tickers ────────────────
# "scale" converts yfinance Close price to model's expected price units.
COMMODITY_TICKERS: dict[str, dict] = {
    "Steel":       {"ticker": "SLX",   "type": "etf_proxy",  "scale": 7.5,
                    "note": "VanEck Steel ETF - scaled to approx USD/tonne"},
    "Aluminum":    {"ticker": "AA",    "type": "etf_proxy",  "scale": 60.0,
                    "note": "Alcoa Corp - scaled to approx LME USD/tonne"},
    "Copper":      {"ticker": "HG=F",  "type": "futures",    "scale": 2204.62,
                    "note": "CME Copper futures - USD/lb converted to USD/tonne"},
    "Platinum":    {"ticker": "PL=F",  "type": "futures",    "scale": 1.0,
                    "note": "NYMEX Platinum futures - USD/troy oz"},
    "Palladium":   {"ticker": "PA=F",  "type": "futures",    "scale": 1.0,
                    "note": "NYMEX Palladium futures - USD/troy oz"},
    "Lithium":     {"ticker": "LIT",   "type": "etf_proxy",  "scale": 0.25,
                    "note": "Global X Lithium ETF - scaled to approx USD/kg"},
    "Natural_Gas": {"ticker": "NG=F",  "type": "futures",    "scale": 10.0,
                    "note": "NYMEX Henry Hub - USD/MMBtu scaled to p/therm equiv"},
    "Nickel":      {"ticker": "VALE",  "type": "etf_proxy",  "scale": 750.0,
                    "note": "Vale SA (major nickel producer) - scaled to approx LME USD/tonne"},
    "Cobalt":      {"ticker": "GLNCY", "type": "etf_proxy",  "scale": 1600.0,
                    "note": "Glencore (top cobalt producer) - scaled to approx USD/tonne"},
}
# Rhodium, Polypropylene, ABS_Resin: no exchange-traded instruments available.
# Synthetic data is generated correlated to oil prices.

MARKET_TICKERS = {
    "SP500":     "^GSPC",
    "DowJones":  "^DJI",
    "VIX":       "^VIX",
    "OilCrude":  "CL=F",
    "Gold":      "GC=F",
    "UST10Y":    "^TNX",
}

MACRO_TICKERS = {
    "oil_price_usd": "CL=F",
    "dxy_index":     "DX-Y.NYB",
    "gold_usd":      "GC=F",
    "ust_10y":       "^TNX",
    "vix":           "^VIX",
    "sp500":         "^GSPC",
}

FX_TICKERS = {
    "usd_gbp": "GBPUSD=X",
    "usd_eur": "EURUSD=X",
    "usd_jpy": "USDJPY=X",
    "usd_cny": "USDCNY=X",
}

CRYPTO_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def _fetch_yfinance_batch(
    ticker_map: dict[str, str],
    period: str,
    interval: str,
    label: str,
) -> pl.DataFrame:
    """Fetch multiple tickers in one yf.download call."""
    import pandas as pd

    all_tickers = list(ticker_map.values())
    ticker_to_name = {v: k for k, v in ticker_map.items()}

    try:
        data = yf.download(all_tickers, period=period, interval=interval, progress=False)
        if data.empty:
            logger.warning(f"No data returned from Yahoo Finance for {label}")
            return pl.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            close_data = data["Close"]
        else:
            close_data = data[["Close"]]

        pdf = close_data.reset_index()
        pdf.columns = ["date"] + [ticker_to_name.get(c, c) for c in close_data.columns]
        df = pl.from_pandas(pdf)

        value_cols = [c for c in df.columns if c != "date"]
        df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in value_cols])
        logger.info(f"Fetched {df.height} rows of {label} from Yahoo Finance")
        return df

    except Exception as e:
        logger.error(f"{label} fetch failed: {e}")
        return pl.DataFrame()


def fetch_commodity_prices(period: str = "7y", interval: str = "1mo") -> pl.DataFrame:
    """
    Fetch real commodity prices from yfinance.

    Returns a Polars DF with columns: date + commodity names.
    Each price is scaled/converted to match the model's expected units.
    """
    import pandas as pd

    logger.info(f"Fetching commodity prices from Yahoo Finance (period={period}, interval={interval})")

    tickers = {info["ticker"] for info in COMMODITY_TICKERS.values()}
    ticker_list = sorted(tickers)

    try:
        raw = yf.download(ticker_list, period=period, interval=interval, progress=False)
        if raw.empty:
            logger.warning("No commodity data returned from Yahoo Finance")
            return pl.DataFrame()
    except Exception as e:
        logger.error(f"Commodity bulk download failed: {e}")
        return pl.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close_data = raw["Close"]
    else:
        close_data = raw[["Close"]]

    dates = close_data.index
    result = {"date": pd.to_datetime(dates)}

    fetched = []
    failed = []
    for commodity, info in COMMODITY_TICKERS.items():
        ticker = info["ticker"]
        scale = info["scale"]
        if ticker in close_data.columns:
            series = close_data[ticker].values.flatten() * scale
            result[commodity] = np.round(series, 2)
            fetched.append(commodity)
        else:
            failed.append(commodity)

    if failed:
        logger.warning(f"Tickers not found in download: {failed}")

    df = pl.from_pandas(pd.DataFrame(result))
    value_cols = [c for c in df.columns if c != "date"]
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in value_cols])

    logger.info(f"Fetched {len(fetched)}/{len(COMMODITY_TICKERS)} commodities: {', '.join(fetched)}")
    if failed:
        logger.info(f"Missing (will use synthetic): {', '.join(failed)}")
    return df


def fetch_macro_from_yfinance(period: str = "7y", interval: str = "1mo") -> pl.DataFrame:
    """
    Fetch macro indicator proxies from yfinance.

    Returns: oil_price_usd, dxy_index, usd_gbp, usd_eur, and more.
    """
    import pandas as pd

    macro_map = {**MACRO_TICKERS, **FX_TICKERS}
    tickers = list(set(macro_map.values()))

    try:
        raw = yf.download(tickers, period=period, interval=interval, progress=False)
        if raw.empty:
            logger.warning("No macro data returned from Yahoo Finance")
            return pl.DataFrame()
    except Exception as e:
        logger.error(f"Macro fetch failed: {e}")
        return pl.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close_data = raw["Close"]
    else:
        close_data = raw[["Close"]]

    result = {"date": pd.to_datetime(close_data.index)}
    ticker_to_name = {v: k for k, v in macro_map.items()}

    for ticker in close_data.columns:
        name = ticker_to_name.get(ticker, ticker)
        result[name] = close_data[ticker].values.flatten()

    df = pl.from_pandas(pd.DataFrame(result))
    value_cols = [c for c in df.columns if c != "date"]
    df = df.with_columns([pl.col(c).forward_fill().alias(c) for c in value_cols])

    logger.info(f"Fetched {len(value_cols)} macro indicators from Yahoo Finance")
    return df


def fetch_market_data(period: str = "7y", interval: str = "1mo") -> pl.DataFrame:
    return _fetch_yfinance_batch(MARKET_TICKERS, period, interval, "market indices")


def fetch_fx_rates(period: str = "7y", interval: str = "1mo") -> pl.DataFrame:
    return _fetch_yfinance_batch(FX_TICKERS, period, interval, "FX rates")


def fetch_single_ticker(ticker: str, period: str = "5y", interval: str = "1d") -> pl.DataFrame:
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
    return {
        "commodities": fetch_commodity_prices(period=period, interval=interval),
        "market_indices": fetch_market_data(period=period, interval=interval),
        "fx_rates": fetch_fx_rates(period=period, interval=interval),
    }


def get_data_source_info() -> dict[str, dict]:
    """Return metadata about each commodity's data source for dashboard/README."""
    info = {}
    for name, meta in COMMODITY_TICKERS.items():
        info[name] = {
            "ticker": meta["ticker"],
            "type": meta["type"],
            "source": "Yahoo Finance",
            "note": meta["note"],
            "is_direct": meta["type"] == "futures",
        }
    for name in ["Rhodium", "Polypropylene", "ABS_Resin"]:
        info[name] = {
            "ticker": None,
            "type": "synthetic",
            "source": "O-U process correlated to oil",
            "note": "No exchange-traded instrument. Synthetic data correlated to energy prices.",
            "is_direct": False,
        }
    return info
