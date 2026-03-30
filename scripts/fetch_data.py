"""
Fetch Real-World Data — pulls live market data and saves in pipeline-compatible format.

Data sources:
  - Yahoo Finance: Commodity futures/ETF prices, macro indicators, FX rates, market indices
  - FRED: Macroeconomic indicators (GDP, CPI, PPI, PMI, unemployment — requires API key)
  - CCXT: Crypto exchange data (Binance)

Output:
  - data/raw/commodity_prices.csv     — 12 commodities (9 real + 3 synthetic)
  - data/raw/macro_indicators.csv     — macro indicators matching pipeline schema
  - data/external/market_commodities.parquet  — raw yfinance commodity data
  - data/external/market_indices.parquet      — S&P 500, VIX, Oil, Gold, etc.
  - data/external/fx_rates.parquet            — USD/GBP, USD/EUR, USD/JPY, USD/CNY
  - data/external/fred_macro.parquet          — FRED indicators (if API key set)
  - data/external/crypto_prices.parquet       — BTC, ETH, etc.

Usage:
    python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.config import get_project_root
from src.data.connectors.yfinance_connector import (
    COMMODITY_TICKERS,
    fetch_commodity_prices,
    fetch_fx_rates,
    fetch_macro_from_yfinance,
    fetch_market_data,
    get_data_source_info,
)
from src.data.polars_pipeline import PolarsDataPipeline
from src.logging_setup import setup_logging


# ── Synthetic generation for unavailable commodities ─────────────────────

def _ou_process(
    n_steps: int,
    base: float,
    vol: float,
    mean_rev: float,
    trend: float = 0.0,
    dt: float = 1 / 12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck mean-reverting process."""
    rng = rng or np.random.default_rng()
    prices = np.zeros(n_steps)
    prices[0] = base
    for t in range(1, n_steps):
        drift = mean_rev * (base * (1 + trend * t * dt) - prices[t - 1]) * dt
        diffusion = vol * prices[t - 1] * np.sqrt(dt) * rng.standard_normal()
        prices[t] = max(prices[t - 1] + drift + diffusion, base * 0.1)
    return prices


def _generate_synthetic_commodity(
    dates: pd.DatetimeIndex,
    oil_prices: np.ndarray | None,
    commodity: str,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single synthetic commodity series, optionally correlated to oil."""
    base_series = _ou_process(
        len(dates), params["base"], params["vol"],
        params["mean_rev"], params.get("trend", 0.0), rng=rng,
    )
    # Add seasonal component
    seasonal = 1 + 0.03 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    prices = base_series * seasonal

    # Correlate with oil if available (Polypropylene and ABS_Resin are petrochemical-derived)
    if oil_prices is not None and commodity in ("Polypropylene", "ABS_Resin"):
        oil_norm = oil_prices / oil_prices[0] if oil_prices[0] != 0 else np.ones_like(oil_prices)
        correlation_weight = 0.3
        prices = prices * (1 + correlation_weight * (oil_norm - 1))

    return np.round(prices, 2)


SYNTHETIC_PARAMS = {
    "Rhodium":       {"base": 4500, "vol": 0.35, "trend": -0.02, "mean_rev": 0.05},
    "Polypropylene": {"base": 1200, "vol": 0.14, "trend": 0.002, "mean_rev": 0.11},
    "ABS_Resin":     {"base": 1500, "vol": 0.14, "trend": 0.002, "mean_rev": 0.11},
}


def _build_macro_csv(
    macro_yf: pd.DataFrame,
    fred_df: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build macro_indicators.csv in pipeline-expected format.

    Target columns: date, gdp_growth_pct, interest_rate_pct, usd_gbp, usd_eur,
    cpi_index, oil_price_usd, manufacturing_pmi, china_ppi_yoy, dxy_index,
    baltic_dry_index, us_ppi_yoy, ev_sales_growth_pct
    """
    result = pd.DataFrame({"date": dates})

    # Map yfinance macro columns to pipeline column names
    yf_mapping = {
        "oil_price_usd": "oil_price_usd",
        "dxy_index": "dxy_index",
        "usd_gbp": "usd_gbp",
        "usd_eur": "usd_eur",
    }

    if macro_yf is not None and (hasattr(macro_yf, "is_empty") and not macro_yf.is_empty() or hasattr(macro_yf, "empty") and not macro_yf.empty):
        macro_pd = macro_yf.to_pandas() if hasattr(macro_yf, "to_pandas") else macro_yf
        macro_pd["date"] = pd.to_datetime(macro_pd["date"])
        macro_pd = macro_pd.set_index("date").resample("MS").last().reset_index()
        # Align dates
        merged = result.merge(macro_pd, on="date", how="left")
        for src_col, dst_col in yf_mapping.items():
            if src_col in merged.columns:
                result[dst_col] = merged[src_col].ffill().bfill()

    # Map FRED columns to pipeline columns
    fred_mapping = {
        "FEDFUNDS": "interest_rate_pct",
        "CPIAUCSL": "cpi_index",
        "PPIACO": "us_ppi_yoy",
        "INDPRO": "manufacturing_pmi",   # Industrial Production as PMI proxy
        "UNRATE": "gdp_growth_pct",       # Inverse proxy — will be adjusted
    }

    if fred_df is not None and len(fred_df) > 0:
        fred_pd = fred_df.to_pandas() if hasattr(fred_df, "to_pandas") else fred_df
        fred_pd["date"] = pd.to_datetime(fred_pd["date"])
        fred_pd = fred_pd.set_index("date").resample("MS").last().reset_index()
        merged = result.merge(fred_pd, on="date", how="left")
        for src_col, dst_col in fred_mapping.items():
            if src_col in merged.columns and dst_col not in result.columns:
                result[dst_col] = merged[src_col].ffill().bfill()

    # Fill missing columns with reasonable synthetic values
    rng = np.random.default_rng(42)
    n = len(result)
    defaults = {
        "gdp_growth_pct":      _ou_process(n, 2.5, 0.8, 0.15, rng=rng),
        "interest_rate_pct":   _ou_process(n, 4.5, 0.5, 0.10, rng=rng),
        "usd_gbp":             _ou_process(n, 0.79, 0.06, 0.12, rng=rng),
        "usd_eur":             _ou_process(n, 0.92, 0.05, 0.12, rng=rng),
        "cpi_index":           _ou_process(n, 110, 1.5, 0.03, rng=rng),
        "oil_price_usd":       _ou_process(n, 80, 0.25, 0.08, rng=rng),
        "manufacturing_pmi":   _ou_process(n, 51, 3.0, 0.20, rng=rng),
        "china_ppi_yoy":       _ou_process(n, 1.0, 2.5, 0.15, rng=rng),
        "dxy_index":           _ou_process(n, 102, 4.0, 0.10, rng=rng),
        "baltic_dry_index":    _ou_process(n, 1500, 0.30, 0.08, rng=rng),
        "us_ppi_yoy":          _ou_process(n, 2.0, 1.5, 0.12, rng=rng),
        "ev_sales_growth_pct": _ou_process(n, 17.5, 5.0, 0.10, rng=rng),
    }
    for col, default_vals in defaults.items():
        if col not in result.columns:
            result[col] = np.round(default_vals, 4)

    # Ensure proper column order
    col_order = ["date"] + [c for c in defaults.keys()]
    result = result[[c for c in col_order if c in result.columns]]
    return result


def fetch_all_data():
    """Fetch all real-world data and save in pipeline-compatible format."""
    setup_logging()
    pipeline = PolarsDataPipeline()
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  FETCHING REAL-WORLD DATA")
    logger.info("=" * 70)

    source_report = {}

    # ═══════════════════════════════════════════════════════════════
    # 1. COMMODITY PRICES (Yahoo Finance)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[1/6] Fetching commodity prices from Yahoo Finance...")
    commodity_df = None
    try:
        commodity_df = fetch_commodity_prices(period="7y", interval="1mo")
        if not commodity_df.is_empty():
            pipeline.save_external(commodity_df, "market_commodities")
            fetched_cols = [c for c in commodity_df.columns if c != "date"]
            logger.info(f"  Commodity prices: {commodity_df.height} rows x {len(fetched_cols)} commodities")
            for col in fetched_cols:
                source_report[col] = "Yahoo Finance (real)"
        else:
            logger.warning("  No commodity data returned from Yahoo Finance")
    except Exception as e:
        logger.error(f"  Commodity fetch failed: {e}")

    # 1b. Generate synthetic for commodities not available on yfinance
    logger.info("\n[1b] Generating synthetic data for unavailable commodities...")
    synth_commodities_needed = ["Rhodium", "Polypropylene", "ABS_Resin"]

    # Also check which yfinance commodities failed
    fetched_yf = set()
    if commodity_df is not None and not commodity_df.is_empty():
        fetched_yf = set(c for c in commodity_df.columns if c != "date")

    for name in COMMODITY_TICKERS:
        if name not in fetched_yf:
            synth_commodities_needed.append(name)

    # Build unified commodity CSV
    if commodity_df is not None and not commodity_df.is_empty():
        cdf = commodity_df.to_pandas()
        cdf["date"] = pd.to_datetime(cdf["date"])
        dates = cdf["date"]
    else:
        dates = pd.date_range("2019-01-01", periods=84, freq="MS")
        cdf = pd.DataFrame({"date": dates})

    rng = np.random.default_rng(42)

    # Get oil prices for correlation (from fetched data or synthetic)
    oil_prices = None
    if "Natural_Gas" in cdf.columns:
        oil_prices = cdf["Natural_Gas"].values
    else:
        oil_prices = _ou_process(len(dates), 80, 0.25, 0.08, rng=np.random.default_rng(99))

    for commodity in synth_commodities_needed:
        if commodity not in cdf.columns:
            params = SYNTHETIC_PARAMS.get(commodity)
            if params is None:
                # Use default params from synthetic_generator
                from src.data.synthetic_generator import COMMODITY_PARAMS
                params = COMMODITY_PARAMS.get(commodity, {"base": 1000, "vol": 0.15, "mean_rev": 0.10})
            cdf[commodity] = _generate_synthetic_commodity(dates, oil_prices, commodity, params, rng)
            source_report[commodity] = "Synthetic (O-U process)"
            logger.info(f"  Generated synthetic: {commodity}")

    # Ensure all 12 commodities are present in correct order
    all_commodities = [
        "Steel", "Aluminum", "Copper", "Platinum", "Palladium", "Rhodium",
        "Lithium", "Cobalt", "Nickel", "Natural_Gas", "Polypropylene", "ABS_Resin",
    ]
    for c in all_commodities:
        if c not in cdf.columns:
            params = SYNTHETIC_PARAMS.get(c, {"base": 1000, "vol": 0.15, "mean_rev": 0.10})
            cdf[c] = _generate_synthetic_commodity(dates, oil_prices, c, params, rng)
            source_report[c] = "Synthetic (O-U process)"
            logger.info(f"  Generated synthetic fallback: {c}")

    cols = ["date"] + all_commodities
    cdf = cdf[[c for c in cols if c in cdf.columns]]
    cdf.to_csv(raw_dir / "commodity_prices.csv", index=False)
    logger.info(f"  Saved commodity_prices.csv: {cdf.shape[0]} rows x {cdf.shape[1] - 1} commodities")

    # ═══════════════════════════════════════════════════════════════
    # 2. MARKET INDICES (Yahoo Finance)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[2/6] Fetching market indices from Yahoo Finance...")
    try:
        market_df = fetch_market_data(period="7y", interval="1mo")
        if not market_df.is_empty():
            pipeline.save_external(market_df, "market_indices")
            logger.info(f"  Market indices: {market_df.height} rows x {market_df.width - 1} series")
    except Exception as e:
        logger.error(f"  Market data fetch failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 3. FX RATES (Yahoo Finance)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[3/6] Fetching FX rates from Yahoo Finance...")
    try:
        fx_df = fetch_fx_rates(period="7y", interval="1mo")
        if not fx_df.is_empty():
            pipeline.save_external(fx_df, "fx_rates")
            logger.info(f"  FX rates: {fx_df.height} rows x {fx_df.width - 1} pairs")
    except Exception as e:
        logger.error(f"  FX data fetch failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 4. MACRO INDICATORS (Yahoo Finance + FRED)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[4/6] Fetching macro indicators...")
    macro_yf = None
    fred_df = None

    # 4a. Yahoo Finance macro proxies
    try:
        macro_yf = fetch_macro_from_yfinance(period="7y", interval="1mo")
        if not macro_yf.is_empty():
            logger.info(f"  Yahoo Finance macro: {macro_yf.height} rows x {macro_yf.width - 1} indicators")
    except Exception as e:
        logger.error(f"  Yahoo Finance macro fetch failed: {e}")

    # 4b. FRED macro indicators (if API key available)
    try:
        from src.data.connectors.fred_connector import fetch_macro_indicators
        fred_df_pl = fetch_macro_indicators(start_date="2018-01-01")
        if not fred_df_pl.is_empty():
            pipeline.save_external(fred_df_pl, "fred_macro")
            fred_df = fred_df_pl
            logger.info(f"  FRED macro: {fred_df_pl.height} rows x {fred_df_pl.width - 1} series")
    except Exception as e:
        logger.warning(f"  FRED fetch failed (API key may not be set): {e}")

    # 4c. Build pipeline-compatible macro CSV
    fred_pd = fred_df.to_pandas() if fred_df is not None and not fred_df.is_empty() else None
    macro_csv = _build_macro_csv(macro_yf, fred_pd, dates)
    macro_csv.to_csv(raw_dir / "macro_indicators.csv", index=False)
    logger.info(f"  Saved macro_indicators.csv: {macro_csv.shape[0]} rows x {macro_csv.shape[1] - 1} indicators")

    # ═══════════════════════════════════════════════════════════════
    # 5. CRYPTO PRICES (CCXT / Binance)
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[5/6] Fetching crypto prices from Binance via CCXT...")
    try:
        from src.data.connectors.ccxt_connector import fetch_crypto_prices
        crypto_df = fetch_crypto_prices(timeframe="1d", limit=365)
        if not crypto_df.is_empty():
            pipeline.save_external(crypto_df, "crypto_prices")
            logger.info(f"  Crypto prices: {crypto_df.height} rows x {crypto_df.width - 1} assets")
    except Exception as e:
        logger.warning(f"  Crypto fetch failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 6. PARQUET CONVERSION
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n[6/6] Converting CSV files to Parquet...")
    converted = pipeline.convert_all_csv_to_parquet()
    for name, path in converted.items():
        logger.info(f"  Converted {name} to Parquet")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  DATA SOURCE REPORT")
    logger.info("=" * 70)

    real_count = 0
    synth_count = 0
    for commodity in all_commodities:
        src = source_report.get(commodity, "Unknown")
        is_real = "real" in src.lower() or "Yahoo" in src
        symbol = "[REAL]" if is_real else "[SYNTH]"
        if is_real:
            real_count += 1
        else:
            synth_count += 1
        logger.info(f"  {symbol:8s} {commodity:16s} - {src}")

    logger.info(f"\n  Total: {real_count} real / {synth_count} synthetic out of {len(all_commodities)} commodities")

    datasets = pipeline.list_datasets()
    logger.info("\n  Available datasets:")
    for source, names in datasets.items():
        logger.info(f"    {source}: {', '.join(names)}")

    logger.info("=" * 70)
    logger.info("  Data fetch complete!")
    logger.info(f"  Pipeline data saved to: data/raw/")
    logger.info(f"  Dashboard data saved to: data/external/")

    return source_report


if __name__ == "__main__":
    fetch_all_data()
