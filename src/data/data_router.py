"""
Data Router — reads config/settings.yaml and returns the correct data source.

Pattern: Every downstream class depends on the Protocol interface, never on the
concrete implementation. To swap data sources, change settings.yaml — no code changes.

Usage:
    from src.data.data_router import get_operational_source, get_market_source

    sales = get_operational_source().get_sales("2024-01-01", "2025-12-31")
    prices = get_market_source().get_commodity_prices("2024-01-01", "2025-12-31")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger

from src.config import get_project_root, get_settings
from src.data.data_source_protocol import MarketDataSource, OperationalDataSource


# ═══════════════════════════════════════════════════════════════════
# OPERATIONAL DATA SOURCES
# ═══════════════════════════════════════════════════════════════════


class SyntheticOperationalSource:
    """
    JLR-calibrated synthetic operational data via Ornstein-Uhlenbeck processes.
    Default source for development, testing, and hackathon demo.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._settings = get_settings()

    def get_sales(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_sales_data
        df = generate_sales_data(seed=self._seed)
        return self._filter_dates(df, from_date, to_date)

    def get_production(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_production_inventory
        df = generate_production_inventory(seed=self._seed)
        return self._filter_dates(df, from_date, to_date)

    def get_cogs_detail(self, from_date: str, to_date: str) -> pd.DataFrame:
        """Approximate COGS detail from BOM weights and commodity prices."""
        bom = self.get_bom()
        sales = self.get_sales(from_date, to_date)
        if sales.empty or bom.empty:
            return pd.DataFrame(columns=["date", "segment", "commodity", "cost_usd", "volume"])
        merged = sales.merge(bom, on="segment", how="left")
        merged["cost_usd"] = (
            merged["volume"] * merged["avg_price_usd"]
            * self._settings["financial"]["base_cogs_pct"]
            * self._settings["financial"]["material_cogs_fraction"]
            * merged["bom_weight"]
        )
        result = merged[["date", "segment", "commodity", "cost_usd", "volume"]].copy()
        return result

    def get_bom(self) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_bom_data
        return generate_bom_data()

    def get_inventory(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_production_inventory
        df = generate_production_inventory(seed=self._seed)
        filtered = self._filter_dates(df, from_date, to_date)
        settings = get_settings()
        avg_price = sum(s["avg_price_usd"] for s in settings["vehicle_segments"]) / len(
            settings["vehicle_segments"]
        )
        filtered = filtered.copy()
        filtered["inventory_value_usd"] = filtered["ending_inventory"] * avg_price * 0.6
        return filtered[["date", "segment", "ending_inventory", "inventory_value_usd"]]

    @staticmethod
    def _filter_dates(df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
        if "date" not in df.columns:
            return df
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(from_date)) & (df["date"] <= pd.to_datetime(to_date))
        return df[mask].reset_index(drop=True)


class ParquetOperationalSource:
    """
    Reads operational data from data/parquet/ or falls back to data/raw/ CSVs.

    Drop a file named sales_data.csv (or sales_data.parquet) into data/raw/ or
    data/parquet/ and this source will pick it up automatically.
    """

    _FILE_MAP = {
        "sales": "sales_data",
        "production": "production_inventory",
        "bom": "bom_data",
    }

    def __init__(self) -> None:
        self._root = get_project_root()

    def get_sales(self, from_date: str, to_date: str) -> pd.DataFrame:
        df = self._load("sales_data")
        return self._filter_dates(df, from_date, to_date)

    def get_production(self, from_date: str, to_date: str) -> pd.DataFrame:
        df = self._load("production_inventory")
        return self._filter_dates(df, from_date, to_date)

    def get_cogs_detail(self, from_date: str, to_date: str) -> pd.DataFrame:
        try:
            df = self._load("cogs_detail")
            return self._filter_dates(df, from_date, to_date)
        except FileNotFoundError:
            # Fall back to synthetic approximation if cogs_detail not present
            logger.warning("cogs_detail not found — synthesising from BOM + sales")
            synth = SyntheticOperationalSource()
            return synth.get_cogs_detail(from_date, to_date)

    def get_bom(self) -> pd.DataFrame:
        return self._load("bom_data")

    def get_inventory(self, from_date: str, to_date: str) -> pd.DataFrame:
        try:
            df = self._load("inventory_data")
        except FileNotFoundError:
            df = self._load("production_inventory")
        filtered = self._filter_dates(df, from_date, to_date)
        if "inventory_value_usd" not in filtered.columns:
            filtered = filtered.copy()
            filtered["inventory_value_usd"] = filtered.get("ending_inventory", 0) * 0
        return filtered

    def _load(self, name: str) -> pd.DataFrame:
        """Try parquet first, then CSV. Raise FileNotFoundError with helpful message."""
        parquet_path = self._root / "data" / "parquet" / f"{name}.parquet"
        raw_parquet = self._root / "data" / "raw" / f"{name}.parquet"
        raw_csv = self._root / "data" / "raw" / f"{name}.csv"
        synth_csv = self._root / "data" / "synthetic" / f"{name}.csv"

        for path in [parquet_path, raw_parquet, raw_csv, synth_csv]:
            if path.exists():
                logger.debug(f"ParquetOperationalSource: loading {path}")
                if path.suffix == ".parquet":
                    return pl.read_parquet(path).to_pandas()
                return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=1).columns else [])

        raise FileNotFoundError(
            f"Cannot find {name} in data/parquet/, data/raw/, or data/synthetic/. "
            f"Run `python scripts/generate_data.py` to generate synthetic data first."
        )

    @staticmethod
    def _filter_dates(df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
        if "date" not in df.columns:
            return df
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(from_date)) & (df["date"] <= pd.to_datetime(to_date))
        return df[mask].reset_index(drop=True)


class SAPOperationalSource:
    """
    SAP S/4HANA connector stub. Ready for production integration.

    Implement using pyrfc (SAP RFC library) or SAP REST API (OData v4).
    See: src/data/connectors/erp_connector.py for connection template.

    Required environment variables:
        SAP_HOST, SAP_USER, SAP_PASS, SAP_CLIENT, SAP_SYSNR

    To activate:
        1. pip install pyrfc (requires SAP NW RFC SDK)
        2. Implement each method below using RFC calls or OData endpoints
        3. Set operational_data_source: sap in config/settings.yaml
    """

    _MSG = (
        "SAP S/4HANA connector not yet implemented. "
        "Set `operational_data_source: parquet` in config/settings.yaml "
        "and provide your data files in data/raw/ or data/parquet/."
    )

    def get_sales(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_production(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_cogs_detail(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_bom(self) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_inventory(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)


class SalesforceOperationalSource:
    """
    Salesforce CRM connector stub (Sales Cloud / Revenue Cloud).

    Implement using simple_salesforce or Salesforce REST API.
    Required env vars: SF_USERNAME, SF_PASSWORD, SF_TOKEN, SF_DOMAIN

    To activate:
        1. pip install simple_salesforce
        2. Implement SOQL queries for each method
        3. Set operational_data_source: salesforce in config/settings.yaml
    """

    _MSG = (
        "Salesforce connector not yet implemented. "
        "Set `operational_data_source: parquet` in config/settings.yaml "
        "and provide your data files in data/raw/ or data/parquet/."
    )

    def get_sales(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_production(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_cogs_detail(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_bom(self) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_inventory(self, from_date: str, to_date: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)


# ═══════════════════════════════════════════════════════════════════
# MARKET DATA SOURCES
# ═══════════════════════════════════════════════════════════════════


class YFinanceMarketSource:
    """
    Live commodity prices and macro indicators from Yahoo Finance + FRED.
    9/12 commodities have exchange-traded proxies; 3 use synthetic.
    """

    def get_commodity_prices(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.connectors.yfinance_connector import fetch_commodity_prices
        df_pl = fetch_commodity_prices()
        if df_pl.is_empty():
            logger.warning("YFinance returned empty — falling back to synthetic commodity prices")
            return _synthetic_market_source().get_commodity_prices(from_date, to_date)
        df = df_pl.to_pandas()
        return _filter_dates_market(df, from_date, to_date)

    def get_macro_indicators(self, from_date: str, to_date: str) -> pd.DataFrame:
        try:
            from src.data.connectors.fred_connector import fetch_macro_indicators
            df_pl = fetch_macro_indicators()
            if df_pl.is_empty():
                raise ValueError("FRED returned empty")
            df = df_pl.to_pandas()
            return _filter_dates_market(df, from_date, to_date)
        except Exception as e:
            logger.warning(f"FRED fetch failed ({e}) — using yfinance macro proxies")
            from src.data.connectors.yfinance_connector import fetch_macro_from_yfinance
            df_pl = fetch_macro_from_yfinance()
            if df_pl.is_empty():
                return _synthetic_market_source().get_macro_indicators(from_date, to_date)
            df = df_pl.to_pandas()
            return _filter_dates_market(df, from_date, to_date)

    def get_fx_rates(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.connectors.yfinance_connector import FX_TICKERS, _fetch_yfinance_batch
        df_pl = _fetch_yfinance_batch(FX_TICKERS, period="7y", interval="1mo", label="FX rates")
        if df_pl.is_empty():
            return _synthetic_market_source().get_fx_rates(from_date, to_date)
        df = df_pl.to_pandas()
        return _filter_dates_market(df, from_date, to_date)


class ParquetMarketSource:
    """
    Reads market data from data/external/ or data/parquet/.
    Falls back to synthetic if files are not present.
    """

    def __init__(self) -> None:
        self._root = get_project_root()

    def get_commodity_prices(self, from_date: str, to_date: str) -> pd.DataFrame:
        df = self._load_external("market_commodities", "commodity_prices")
        return _filter_dates_market(df, from_date, to_date)

    def get_macro_indicators(self, from_date: str, to_date: str) -> pd.DataFrame:
        df = self._load_external("fred_macro", "macro_indicators")
        return _filter_dates_market(df, from_date, to_date)

    def get_fx_rates(self, from_date: str, to_date: str) -> pd.DataFrame:
        try:
            df = self._load_external("fx_rates", "macro_indicators")
        except FileNotFoundError:
            df = self.get_macro_indicators(from_date, to_date)
            fx_cols = ["date"] + [c for c in df.columns if "usd" in c.lower()]
            df = df[fx_cols] if len(fx_cols) > 1 else df
        return _filter_dates_market(df, from_date, to_date)

    def _load_external(self, primary: str, fallback: str) -> pd.DataFrame:
        candidates = [
            self._root / "data" / "external" / f"{primary}.parquet",
            self._root / "data" / "parquet" / f"{primary}.parquet",
            self._root / "data" / "external" / f"{fallback}.parquet",
            self._root / "data" / "raw" / f"{fallback}.csv",
            self._root / "data" / "synthetic" / f"{fallback}.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    return pl.read_parquet(path).to_pandas()
                return pd.read_csv(path)
        raise FileNotFoundError(
            f"Cannot find {primary} or {fallback} in data/. "
            "Run `python scripts/fetch_data.py` for live data or "
            "`python scripts/generate_data.py` for synthetic."
        )


def _synthetic_market_source():
    """Helper: returns a SyntheticMarketSource (avoids circular import)."""
    return _SyntheticMarketSource()


class _SyntheticMarketSource:
    """Internal fallback when live/parquet market data is unavailable."""

    def get_commodity_prices(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_commodity_prices
        df = generate_commodity_prices()
        return _filter_dates_market(df, from_date, to_date)

    def get_macro_indicators(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_macro_indicators
        df = generate_macro_indicators()
        return _filter_dates_market(df, from_date, to_date)

    def get_fx_rates(self, from_date: str, to_date: str) -> pd.DataFrame:
        from src.data.synthetic_generator import generate_macro_indicators
        df = generate_macro_indicators()
        fx_cols = ["date"] + [c for c in df.columns if "usd" in c.lower()]
        return _filter_dates_market(df[fx_cols], from_date, to_date)


# ═══════════════════════════════════════════════════════════════════
# DATA ROUTER — factory functions keyed off settings.yaml
# ═══════════════════════════════════════════════════════════════════


def get_operational_source(settings: dict[str, Any] | None = None) -> OperationalDataSource:
    """
    Return the configured OperationalDataSource.

    Reads `operational_data_source` from settings.yaml.
    Override by passing a settings dict (useful for tests).
    """
    cfg = settings or get_settings()
    source = cfg.get("operational_data_source", "synthetic")
    match source:
        case "synthetic":
            logger.debug("DataRouter: OperationalDataSource = SyntheticOperationalSource")
            return SyntheticOperationalSource()
        case "parquet":
            logger.debug("DataRouter: OperationalDataSource = ParquetOperationalSource")
            return ParquetOperationalSource()
        case "sap":
            logger.debug("DataRouter: OperationalDataSource = SAPOperationalSource (stub)")
            return SAPOperationalSource()
        case "salesforce":
            logger.debug("DataRouter: OperationalDataSource = SalesforceOperationalSource (stub)")
            return SalesforceOperationalSource()
        case _:
            raise ValueError(
                f"Unknown operational_data_source: '{source}'. "
                "Valid values: synthetic | parquet | sap | salesforce"
            )


def get_market_source(settings: dict[str, Any] | None = None) -> MarketDataSource:
    """
    Return the configured MarketDataSource.

    Reads `market_data_source` from settings.yaml.
    """
    cfg = settings or get_settings()
    source = cfg.get("market_data_source", "live")
    match source:
        case "live":
            logger.debug("DataRouter: MarketDataSource = YFinanceMarketSource")
            return YFinanceMarketSource()
        case "parquet":
            logger.debug("DataRouter: MarketDataSource = ParquetMarketSource")
            return ParquetMarketSource()
        case "synthetic":
            logger.debug("DataRouter: MarketDataSource = _SyntheticMarketSource")
            return _SyntheticMarketSource()
        case _:
            raise ValueError(
                f"Unknown market_data_source: '{source}'. "
                "Valid values: live | parquet | synthetic"
            )


# ── Convenience helpers ───────────────────────────────────────────


def _filter_dates_market(df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
    """Filter a market DataFrame by date range. No-op if date column missing."""
    date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if date_col is None:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= pd.to_datetime(from_date)) & (df[date_col] <= pd.to_datetime(to_date))
    return df[mask].reset_index(drop=True)
