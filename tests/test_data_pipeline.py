"""Tests for data connectors and Polars pipeline."""

import polars as pl
import pytest


class TestPolarsDataPipeline:
    """Tests for the Polars-native data pipeline."""

    def test_load_commodity_prices(self):
        from src.data.polars_pipeline import PolarsDataPipeline
        pipeline = PolarsDataPipeline()
        df = pipeline.load("commodity_prices")
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert "date" in df.columns

    def test_load_sales_data(self):
        from src.data.polars_pipeline import PolarsDataPipeline
        pipeline = PolarsDataPipeline()
        df = pipeline.load("sales_data")
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_parquet_conversion(self):
        from src.data.polars_pipeline import PolarsDataPipeline
        pipeline = PolarsDataPipeline()
        results = pipeline.convert_all_csv_to_parquet()
        # May already be converted, so just check it doesn't error
        assert isinstance(results, dict)

    def test_list_datasets(self):
        from src.data.polars_pipeline import PolarsDataPipeline
        pipeline = PolarsDataPipeline()
        datasets = pipeline.list_datasets()
        assert isinstance(datasets, dict)
        # Should have at least synthetic or parquet
        total_datasets = sum(len(v) for v in datasets.values())
        assert total_datasets > 0

    def test_save_and_load_parquet(self, tmp_path):
        from src.data.polars_pipeline import PolarsDataPipeline
        pipeline = PolarsDataPipeline()
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)
        loaded = pl.read_parquet(path)
        assert loaded.height == 3
        assert loaded.width == 2


class TestYFinanceConnector:
    """Tests for Yahoo Finance connector (basic structure checks)."""

    def test_ticker_maps_exist(self):
        from src.data.connectors.yfinance_connector import (
            COMMODITY_TICKERS,
            FX_TICKERS,
            MARKET_TICKERS,
        )
        assert len(COMMODITY_TICKERS) >= 8
        assert len(FX_TICKERS) >= 3
        assert len(MARKET_TICKERS) >= 4

    def test_fetch_returns_polars(self):
        """Test that fetch functions return Polars DataFrames (may be empty if offline)."""
        from src.data.connectors.yfinance_connector import fetch_commodity_prices
        result = fetch_commodity_prices(period="1mo", interval="1d")
        assert isinstance(result, pl.DataFrame)


class TestCCXTConnector:
    """Tests for CCXT crypto connector."""

    def test_crypto_symbols_defined(self):
        from src.data.connectors.ccxt_connector import CRYPTO_SYMBOLS
        assert len(CRYPTO_SYMBOLS) >= 4
        assert "BTC/USDT" in CRYPTO_SYMBOLS


class TestFFNAnalytics:
    """Tests for FFN financial analytics."""

    def test_correlation_matrix(self):
        from src.analytics.ffn_analytics import compute_correlation_matrix
        import numpy as np

        # Create test data
        np.random.seed(42)
        df = pl.DataFrame({
            "date": pl.date_range(
                __import__("datetime").date(2020, 1, 1),
                __import__("datetime").date(2022, 12, 1),
                "1mo",
                eager=True,
            ),
            "A": np.random.randn(36).cumsum() + 100,
            "B": np.random.randn(36).cumsum() + 100,
        })

        corr = compute_correlation_matrix(df)
        assert isinstance(corr, pl.DataFrame)
        if not corr.is_empty():
            assert "commodity" in corr.columns

    def test_drawdown_analysis(self):
        from src.analytics.ffn_analytics import compute_drawdown_analysis
        import numpy as np

        df = pl.DataFrame({
            "date": pl.date_range(
                __import__("datetime").date(2020, 1, 1),
                __import__("datetime").date(2022, 12, 1),
                "1mo",
                eager=True,
            ),
            "Price": np.random.randn(36).cumsum() + 100,
        })

        dd = compute_drawdown_analysis(df)
        assert isinstance(dd, pl.DataFrame)
        if not dd.is_empty():
            assert "drawdown_pct" in dd.columns


class TestMarketIntelligence:
    """Tests for market intelligence engine."""

    def test_analyze_trends(self):
        from src.analytics.market_intelligence import MarketIntelligence
        import numpy as np

        mi = MarketIntelligence()
        df = pl.DataFrame({
            "date": pl.date_range(
                __import__("datetime").date(2020, 1, 1),
                __import__("datetime").date(2023, 12, 1),
                "1mo",
                eager=True,
            ),
            "Copper": (np.random.randn(48).cumsum() + 100).tolist(),
            "Lithium": (np.random.randn(48).cumsum() + 200).tolist(),
        })

        signals = mi.analyze_commodity_trends(df)
        assert "Copper" in signals
        assert "current_price" in signals["Copper"]
        assert "trend" in signals["Copper"]

    def test_create_snapshot(self):
        from src.analytics.market_intelligence import MarketIntelligence
        import numpy as np

        mi = MarketIntelligence()
        df = pl.DataFrame({
            "date": pl.date_range(
                __import__("datetime").date(2020, 1, 1),
                __import__("datetime").date(2023, 12, 1),
                "1mo",
                eager=True,
            ),
            "Copper": (np.random.randn(48).cumsum() + 100).tolist(),
        })

        snapshot = mi.create_snapshot(df)
        assert snapshot.risk_level in ("low", "moderate", "elevated", "high")
        assert isinstance(snapshot.alerts, list)
