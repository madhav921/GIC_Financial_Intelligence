"""
Commodity market data connector — commodities-api.com + FRED fallback.

Primary provider: commodities-api.com (free tier: 250 req/month)
  - API key: set COMMODITIES_API_KEY in .env
  - Docs: https://commodities-api.com/documentation

Supported symbols (subset):
  CRUDE_OIL, BRENT_CRUDE, NATURAL_GAS, COPPER, ALUMINUM, NICKEL,
  ZINC, LEAD, GOLD, SILVER, PLATINUM, PALLADIUM, LITHIUM

Fallback: If API key not set, returns data from yfinance cache.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger

from src.config import get_env, get_project_root


# Mapping: internal commodity names -> commodities-api symbol codes
COMMODITY_API_SYMBOLS: dict[str, str] = {
    "Lithium":   "LITHIUM",
    "Cobalt":    "COBALT",
    "Nickel":    "NICKEL",
    "Aluminum":  "ALU",
    "Steel":     "STEEL",
    "Copper":    "COPPER",
    "Platinum":  "PTLM",
    "Gold":      "XAU",
    "Silver":    "XAG",
    "CrudeOil":  "CRUDEOIL",
    "BrentCrude": "BRENT",
    "NaturalGas": "NG",
}


class CommodityAPIConnector:
    """
    Interface for fetching commodity market data from commodities-api.com.

    Gracefully falls back to cached yfinance data when API key is absent
    or the API returns an error — so the pipeline never hard-fails.

    Usage:
        connector = CommodityAPIConnector()
        df = connector.fetch_time_series("Lithium", start="2022-01-01", end="2024-12-31")
    """

    BASE_URL = "https://commodities-api.com/api"

    def __init__(self):
        self._api_key: str | None = get_env("COMMODITIES_API_KEY")
        if not self._api_key:
            logger.warning(
                "COMMODITIES_API_KEY not set — using yfinance cache as fallback. "
                "Add key to .env to enable live commodities-api data."
            )

    @property
    def _client(self):
        """Lazy-load the commodities_api client (only when key is available)."""
        if self._api_key:
            try:
                from commodities_api.client import CommoditiesApiClient
                return CommoditiesApiClient(self._api_key)
            except ImportError:
                logger.error("commodities_api package not installed. Run: pip install commodities-api")
        return None

    def _symbol(self, commodity: str) -> str | None:
        """Resolve commodity name to API symbol code."""
        symbol = COMMODITY_API_SYMBOLS.get(commodity)
        if symbol is None:
            # Try case-insensitive match
            for k, v in COMMODITY_API_SYMBOLS.items():
                if k.lower() == commodity.lower():
                    return v
        return symbol

    def fetch_latest(
        self, commodities: list[str] | None = None, base: str = "USD"
    ) -> pd.DataFrame:
        """
        Fetch latest commodity prices.

        Args:
            commodities: list of internal commodity names (e.g. ["Lithium", "Copper"])
            base: base currency (default USD)

        Returns:
            pandas DataFrame with columns: symbol, commodity, price, currency, date
        """
        if commodities is None:
            commodities = list(COMMODITY_API_SYMBOLS.keys())

        client = self._client
        if client is None:
            return self._fallback_latest(commodities)

        symbols = [self._symbol(c) for c in commodities if self._symbol(c)]
        if not symbols:
            return pd.DataFrame()

        try:
            result = client.get_latest(base=base, symbols=symbols)
            rates = result.get("data", {}).get("rates", {})
            today = datetime.today().strftime("%Y-%m-%d")
            rows = []
            for commodity, sym in zip(commodities, symbols):
                price = rates.get(sym)
                if price and price > 0:
                    rows.append({
                        "commodity": commodity,
                        "symbol": sym,
                        "price": 1.0 / price if base == "USD" else float(price),
                        "currency": base,
                        "date": today,
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"commodities-api get_latest failed: {e}")
            return self._fallback_latest(commodities)

    def fetch_historical(
        self, commodity: str, fetch_date: str, base: str = "USD"
    ) -> float | None:
        """
        Fetch a single commodity price for a specific historical date.

        Args:
            commodity: internal name (e.g. "Lithium")
            fetch_date: ISO date string "YYYY-MM-DD"
            base: currency

        Returns:
            float price or None if unavailable
        """
        sym = self._symbol(commodity)
        if not sym:
            logger.warning(f"No API symbol for {commodity}")
            return None

        client = self._client
        if client is None:
            return None

        try:
            result = client.get_historical(date=fetch_date, base=base, symbols=[sym])
            rates = result.get("data", {}).get("rates", {})
            raw = rates.get(sym)
            if raw and raw > 0:
                return 1.0 / raw  # API returns currency units per commodity unit
            return None
        except Exception as e:
            logger.error(f"commodities-api get_historical failed for {commodity} on {fetch_date}: {e}")
            return None

    def fetch_time_series(
        self,
        commodity: str,
        start: str,
        end: str | None = None,
        base: str = "USD",
    ) -> pd.DataFrame:
        """
        Fetch historical time series for a commodity.

        Args:
            commodity: internal name (e.g. "Copper")
            start: ISO date "YYYY-MM-DD"
            end: ISO date "YYYY-MM-DD" (defaults to today)
            base: currency

        Returns:
            pandas DataFrame with columns: date, commodity, price
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        sym = self._symbol(commodity)
        if not sym:
            logger.warning(f"No API symbol for {commodity}, using yfinance fallback")
            return self._fallback_time_series(commodity, start, end)

        client = self._client
        if client is None:
            return self._fallback_time_series(commodity, start, end)

        try:
            result = client.get_time_series(
                start_date=start, end_date=end, base=base, symbols=[sym]
            )
            rates_by_date = result.get("data", {}).get("rates", {})

            rows = []
            for date_str, rate_dict in rates_by_date.items():
                raw = rate_dict.get(sym)
                if raw and raw > 0:
                    rows.append({
                        "date": pd.to_datetime(date_str),
                        "commodity": commodity,
                        "price": 1.0 / raw,  # invert to get $/unit
                    })

            if not rows:
                logger.warning(f"No data returned for {commodity}; using fallback")
                return self._fallback_time_series(commodity, start, end)

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            logger.info(f"commodities-api: fetched {len(df)} rows for {commodity}")
            return df

        except Exception as e:
            logger.error(f"commodities-api fetch_time_series failed for {commodity}: {e}")
            return self._fallback_time_series(commodity, start, end)

    def fetch_all_commodities_series(
        self,
        start: str,
        end: str | None = None,
        commodities: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Fetch time series for all configured commodities and return a
        wide-format Polars DataFrame (date + one column per commodity).

        This supplements the yfinance data with more granular spot prices
        when the API key is available.
        """
        if commodities is None:
            commodities = list(COMMODITY_API_SYMBOLS.keys())
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        all_dfs = []
        for commodity in commodities:
            df = self.fetch_time_series(commodity, start, end)
            if not df.empty:
                all_dfs.append(df.pivot(index="date", columns=None, values="price").rename(
                    columns={"price": commodity}
                ))

        if not all_dfs:
            logger.warning("No commodities-api data returned — returning empty DataFrame")
            return pl.DataFrame()

        # Merge all on date
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.join(df, on="date", how="outer")

        result = result.reset_index().sort_values("date")
        return pl.from_pandas(result)

    def save_to_cache(self, df: pl.DataFrame, filename: str = "commodities_api_prices.parquet") -> None:
        """Save fetched data to the external data cache directory."""
        path = get_project_root() / "data" / "external" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        logger.info(f"Saved commodities-api data to {path}")

    # ─── Fallback helpers ─────────────────────────────────────────────────────

    def _fallback_latest(self, commodities: list[str]) -> pd.DataFrame:
        """Return most recent prices from cached yfinance data."""
        path = get_project_root() / "data" / "external" / "market_commodities.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pl.read_parquet(path).to_pandas()
            if "date" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
            latest = df.iloc[-1]
            rows = []
            today = datetime.today().strftime("%Y-%m-%d")
            for c in commodities:
                if c in latest.index:
                    rows.append({
                        "commodity": c,
                        "symbol": self._symbol(c) or "N/A",
                        "price": float(latest[c]),
                        "currency": "USD",
                        "date": today,
                        "source": "yfinance_cache",
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Fallback latest prices failed: {e}")
            return pd.DataFrame()

    def _fallback_time_series(
        self, commodity: str, start: str, end: str
    ) -> pd.DataFrame:
        """Return historical prices from yfinance cache for a single commodity."""
        path = get_project_root() / "data" / "external" / "market_commodities.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pl.read_parquet(path).to_pandas()
            if "date" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"]))
            if commodity not in df.columns:
                return pd.DataFrame()
            series = df[[commodity]].loc[start:end].dropna()
            series = series.reset_index().rename(columns={"index": "date", commodity: "price"})
            series["commodity"] = commodity
            return series[["date", "commodity", "price"]]
        except Exception as e:
            logger.error(f"Fallback time series failed for {commodity}: {e}")
            return pd.DataFrame()

