"""
Fetch real-world data — pulls live data from all configured sources.

Sources:
  - Yahoo Finance: Commodity prices, market indices, FX rates
  - CCXT: Crypto exchange data (Binance)
  - FRED: Macroeconomic indicators (requires API key)

Saves all data as Parquet in data/external/ for fast reload.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.data.connectors.yfinance_connector import (
    fetch_commodity_prices,
    fetch_fx_rates,
    fetch_market_data,
)
from src.data.connectors.ccxt_connector import fetch_crypto_prices
from src.data.connectors.fred_connector import fetch_macro_indicators
from src.data.polars_pipeline import PolarsDataPipeline


def fetch_all_data():
    """Fetch and save all real-world data sources."""
    pipeline = PolarsDataPipeline()

    logger.info("=" * 70)
    logger.info("  FETCHING REAL-WORLD DATA")
    logger.info("=" * 70)

    # ── 1. Yahoo Finance: Commodities ──
    logger.info("\n[1/5] Fetching commodity prices from Yahoo Finance...")
    try:
        commodities_df = fetch_commodity_prices(period="7y", interval="1mo")
        if not commodities_df.is_empty():
            pipeline.save_external(commodities_df, "market_commodities")
            logger.info(f"  ✓ Commodity prices: {commodities_df.height} rows × {commodities_df.width - 1} series")
        else:
            logger.warning("  ✗ No commodity data returned")
    except Exception as e:
        logger.error(f"  ✗ Commodity fetch failed: {e}")

    # ── 2. Yahoo Finance: Market Indices ──
    logger.info("\n[2/5] Fetching market indices from Yahoo Finance...")
    try:
        market_df = fetch_market_data(period="7y", interval="1mo")
        if not market_df.is_empty():
            pipeline.save_external(market_df, "market_indices")
            logger.info(f"  ✓ Market indices: {market_df.height} rows × {market_df.width - 1} series")
        else:
            logger.warning("  ✗ No market data returned")
    except Exception as e:
        logger.error(f"  ✗ Market data fetch failed: {e}")

    # ── 3. Yahoo Finance: FX Rates ──
    logger.info("\n[3/5] Fetching FX rates from Yahoo Finance...")
    try:
        fx_df = fetch_fx_rates(period="7y", interval="1mo")
        if not fx_df.is_empty():
            pipeline.save_external(fx_df, "fx_rates")
            logger.info(f"  ✓ FX rates: {fx_df.height} rows × {fx_df.width - 1} pairs")
        else:
            logger.warning("  ✗ No FX data returned")
    except Exception as e:
        logger.error(f"  ✗ FX data fetch failed: {e}")

    # ── 4. CCXT: Crypto Prices ──
    logger.info("\n[4/5] Fetching crypto prices from Binance via CCXT...")
    try:
        crypto_df = fetch_crypto_prices(timeframe="1d", limit=365)
        if not crypto_df.is_empty():
            pipeline.save_external(crypto_df, "crypto_prices")
            logger.info(f"  ✓ Crypto prices: {crypto_df.height} rows × {crypto_df.width - 1} assets")
        else:
            logger.warning("  ✗ No crypto data returned (ccxt may not be available or rate-limited)")
    except Exception as e:
        logger.error(f"  ✗ Crypto fetch failed: {e}")

    # ── 5. FRED: Macro Indicators ──
    logger.info("\n[5/5] Fetching macroeconomic data from FRED...")
    try:
        fred_df = fetch_macro_indicators(start_date="2018-01-01")
        if not fred_df.is_empty():
            pipeline.save_external(fred_df, "fred_macro")
            logger.info(f"  ✓ FRED macro: {fred_df.height} rows × {fred_df.width - 1} indicators")
        else:
            logger.warning("  ✗ No FRED data (set FRED_API_KEY in .env for macro data)")
    except Exception as e:
        logger.error(f"  ✗ FRED fetch failed: {e}")

    # ── Convert existing CSV to Parquet ──
    logger.info("\n[+] Converting existing CSV files to Parquet...")
    converted = pipeline.convert_all_csv_to_parquet()
    for name, path in converted.items():
        logger.info(f"  ✓ Converted {name} → Parquet")

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    datasets = pipeline.list_datasets()
    for source, names in datasets.items():
        logger.info(f"  {source}: {', '.join(names)}")
    logger.info("=" * 70)
    logger.info("Data fetch complete!")


if __name__ == "__main__":
    fetch_all_data()
