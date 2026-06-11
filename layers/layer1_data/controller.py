"""
Layer 1: Data Architecture Controller

Single entry point for all data acquisition, transformation, and routing.
Delegates to src/data/ modules.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from src.config import get_project_root
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.feature_engineering import prepare_commodity_features


class DataLayerController:
    """
    Layer 1 Controller — Data Architecture

    Usage:
        data = DataLayerController()
        commodity_df = data.load_commodity_data()
        macro_df     = data.load_macro_data()
        sales_df     = data.load_sales_data()
        features_df  = data.build_feature_matrix(commodity_df, macro_df, "Copper")
    """

    def __init__(self):
        self.root = get_project_root()
        self._generator = SyntheticDataGenerator()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_commodity_data(self) -> pd.DataFrame:
        """Load commodity prices. Prefers real yfinance data, falls back to synthetic."""
        return self._load_with_fallback(
            real_paths=[
                self.root / "data" / "raw" / "commodity_prices.csv",
                self.root / "data" / "synthetic" / "commodity_prices.csv",
            ],
            parquet_path=self.root / "data" / "external" / "market_commodities.parquet",
            dataset_name="commodity_prices",
        )

    def load_macro_data(self) -> Optional[pd.DataFrame]:
        """Load macroeconomic indicators (FRED + synthetic)."""
        return self._load_with_fallback(
            real_paths=[
                self.root / "data" / "raw" / "macro_indicators.csv",
                self.root / "data" / "synthetic" / "macro_indicators.csv",
            ],
            parquet_path=self.root / "data" / "external" / "fred_macro.parquet",
            dataset_name="macro_indicators",
        )

    def load_sales_data(self) -> pd.DataFrame:
        """Load vehicle sales data by segment."""
        return self._load_with_fallback(
            real_paths=[self.root / "data" / "synthetic" / "sales_data.csv"],
            parquet_path=None,
            dataset_name="sales_data",
        )

    def load_production_data(self) -> pd.DataFrame:
        """Load production & inventory data."""
        return self._load_with_fallback(
            real_paths=[self.root / "data" / "synthetic" / "production_inventory.csv"],
            parquet_path=None,
            dataset_name="production_inventory",
        )

    def load_bom_data(self) -> pd.DataFrame:
        """Load Bill of Materials commodity weights."""
        return self._load_with_fallback(
            real_paths=[self.root / "data" / "synthetic" / "bom_data.csv"],
            parquet_path=None,
            dataset_name="bom_data",
        )

    def generate_synthetic_data(self) -> dict[str, pd.DataFrame]:
        """Generate all synthetic datasets using Ornstein-Uhlenbeck process."""
        logger.info("Layer 1: Generating synthetic data (O-U process)")
        return {
            "commodity_prices": self._generator.generate_commodity_prices(),
            "macro_indicators": self._generator.generate_macro_indicators(),
            "sales_data": self._generator.generate_sales_data(),
            "production_inventory": self._generator.generate_production_inventory(),
            "bom_data": self._generator.generate_bom_data(),
        }

    def build_feature_matrix(
        self,
        commodity_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame],
        commodity: str,
        lags: list[int] = [1, 3, 6, 12],
    ) -> pd.DataFrame:
        """
        Engineer features for ML training.
        Returns: lags, rolling stats, momentum, macro context, calendar encoding.
        """
        logger.info(f"Layer 1: Building feature matrix for {commodity}")
        return prepare_commodity_features(commodity_df, macro_df, commodity, lags=lags)

    def load_all(self) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """Load all datasets in one call. Returns (commodity_df, macro_df, sales_df, bom_df)."""
        logger.info("Layer 1: Loading all datasets")
        commodity_df = self.load_commodity_data()
        macro_df = self.load_macro_data()
        sales_df = self.load_sales_data()
        bom_df = self.load_bom_data()
        logger.info(
            f"Layer 1: Loaded — commodity={commodity_df.shape}, "
            f"macro={macro_df.shape if macro_df is not None else 'None'}, "
            f"sales={sales_df.shape}"
        )
        return commodity_df, macro_df, sales_df, bom_df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_with_fallback(
        self,
        real_paths: list[Path],
        parquet_path: Optional[Path],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Priority chain: Parquet → real CSV → fallback CSV → synthetic."""
        # 1. Try Parquet (fastest)
        if parquet_path and parquet_path.exists():
            try:
                import polars as pl
                df = pl.read_parquet(parquet_path).to_pandas()
                logger.info(f"Layer 1 [{dataset_name}]: Loaded from Parquet {parquet_path.name}")
                return df
            except Exception as e:
                logger.warning(f"Layer 1 [{dataset_name}]: Parquet load failed — {e}")

        # 2. Try CSV paths in order
        for path in real_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    logger.info(f"Layer 1 [{dataset_name}]: Loaded from CSV {path.name}")
                    return df
                except Exception as e:
                    logger.warning(f"Layer 1 [{dataset_name}]: CSV load failed — {e}")

        # 3. Generate synthetic
        logger.warning(f"Layer 1 [{dataset_name}]: No real data found — generating synthetic")
        datasets = self.generate_synthetic_data()
        return datasets.get(dataset_name, pd.DataFrame())
