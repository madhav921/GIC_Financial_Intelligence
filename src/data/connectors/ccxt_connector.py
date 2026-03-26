"""
CCXT Crypto Exchange Connector — real-time crypto commodity data.

Connects to centralized exchanges (Binance, Coinbase, Kraken, etc.)
for crypto-commodity correlations and digital asset tracking.

Used for:
  - Crypto market sentiment as macro signal
  - BTC/ETH as alternative asset class correlation
  - DeFi commodity tokens tracking
  - Exchange rate arbitrage signals
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
from loguru import logger

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed — crypto data unavailable")


# ── Crypto symbols relevant to commodity/financial analysis ──────
CRYPTO_SYMBOLS = {
    "BTC/USDT":  "Bitcoin",
    "ETH/USDT":  "Ethereum",
    "SOL/USDT":  "Solana",
    "XRP/USDT":  "Ripple",
    "BNB/USDT":  "BNB",
    "AVAX/USDT": "Avalanche",
}

# Timeframe mapping
TIMEFRAMES = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}


def _get_exchange(exchange_id: str = "binance") -> "ccxt.Exchange | None":
    """Initialize exchange connection (public data only — no auth needed)."""
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        return exchange
    except Exception as e:
        logger.error(f"Failed to initialize {exchange_id}: {e}")
        return None


def fetch_crypto_ohlcv(
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    limit: int = 365,
    exchange_id: str = "binance",
) -> pl.DataFrame:
    """
    Fetch OHLCV data for crypto assets from a centralized exchange.

    Args:
        symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
        timeframe: Candle timeframe (1h, 4h, 1d, 1w, 1M)
        limit: Number of candles to fetch per symbol
        exchange_id: Exchange to query (binance, coinbase, kraken, etc.)

    Returns:
        Polars DataFrame with columns: date, symbol, open, high, low, close, volume
    """
    if not CCXT_AVAILABLE:
        logger.warning("ccxt not available, returning empty DataFrame")
        return pl.DataFrame()

    exchange = _get_exchange(exchange_id)
    if exchange is None:
        return pl.DataFrame()

    symbols = symbols or list(CRYPTO_SYMBOLS.keys())
    all_records = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            for candle in ohlcv:
                all_records.append({
                    "date": datetime.fromtimestamp(candle[0] / 1000),
                    "symbol": symbol.replace("/USDT", ""),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                })
            logger.info(f"Fetched {len(ohlcv)} candles for {symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from {exchange_id}: {e}")

    if not all_records:
        return pl.DataFrame()

    return pl.DataFrame(all_records)


def fetch_crypto_prices(
    timeframe: str = "1d",
    limit: int = 365,
    exchange_id: str = "binance",
) -> pl.DataFrame:
    """
    Fetch crypto closing prices in wide format (like commodity prices).

    Returns:
        Polars DataFrame with date + one column per crypto asset
    """
    df = fetch_crypto_ohlcv(timeframe=timeframe, limit=limit, exchange_id=exchange_id)
    if df.is_empty():
        return df

    # Pivot to wide format: date rows × crypto columns
    wide = df.pivot(
        on="symbol",
        index="date",
        values="close",
    ).sort("date")

    return wide


def fetch_exchange_tickers(exchange_id: str = "binance") -> pl.DataFrame:
    """Fetch current spot prices for all tracked crypto assets."""
    if not CCXT_AVAILABLE:
        return pl.DataFrame()

    exchange = _get_exchange(exchange_id)
    if exchange is None:
        return pl.DataFrame()

    records = []
    for symbol, name in CRYPTO_SYMBOLS.items():
        try:
            ticker = exchange.fetch_ticker(symbol)
            records.append({
                "symbol": symbol.replace("/USDT", ""),
                "name": name,
                "last_price": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "volume_24h": ticker.get("quoteVolume"),
                "change_24h_pct": ticker.get("percentage"),
                "high_24h": ticker.get("high"),
                "low_24h": ticker.get("low"),
                "timestamp": datetime.fromtimestamp(ticker["timestamp"] / 1000) if ticker.get("timestamp") else None,
            })
        except Exception as e:
            logger.error(f"Failed to fetch ticker {symbol}: {e}")

    return pl.DataFrame(records) if records else pl.DataFrame()


def get_available_exchanges() -> list[str]:
    """List all exchanges supported by ccxt."""
    if not CCXT_AVAILABLE:
        return []
    return ccxt.exchanges
