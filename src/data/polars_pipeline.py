"""
Polars-native data pipeline — replaces pandas CSV with Polars Parquet.

Provides:
  - Parquet read/write for all datasets (10x faster, 4x smaller)
  - Lazy evaluation for large dataset processing
  - Unified interface for both real-world and synthetic data
  - Auto-conversion from legacy CSV to Parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from src.config import get_project_root, get_settings


class PolarsDataPipeline:
    """High-performance data pipeline using Polars + Parquet."""

    def __init__(self):
        self.settings = get_settings()
        self.root = get_project_root()
        self._raw_dir = self.root / "data" / "raw"
        self._processed_dir = self.root / "data" / "processed"
        self._synthetic_dir = self.root / "data" / "synthetic"
        self._parquet_dir = self.root / "data" / "parquet"
        self._external_dir = self.root / "data" / "external"

        # Ensure directories exist
        for d in [self._raw_dir, self._processed_dir, self._parquet_dir, self._external_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _resolve_parquet(self, name: str) -> Path | None:
        """Find a parquet file by name."""
        path = self._parquet_dir / f"{name}.parquet"
        if path.exists():
            return path
        return None

    def _resolve_csv(self, name: str) -> Path | None:
        """Find a CSV file (raw → synthetic fallback)."""
        for d in [self._raw_dir, self._synthetic_dir]:
            path = d / f"{name}.csv"
            if path.exists():
                return path
        return None

    def save_parquet(self, df: pl.DataFrame, name: str) -> Path:
        """Save a Polars DataFrame to Parquet."""
        path = self._parquet_dir / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
        logger.info(f"Saved {name}.parquet ({df.height} rows, {path.stat().st_size / 1024:.1f} KB)")
        return path

    def save_external(self, df: pl.DataFrame, name: str) -> Path:
        """Save external (real-world) data to Parquet."""
        path = self._external_dir / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
        logger.info(f"Saved external/{name}.parquet ({df.height} rows)")
        return path

    def load(self, name: str) -> pl.DataFrame:
        """
        Load a dataset by name. Priority: parquet → external → raw CSV → synthetic CSV.

        Automatically converts CSV to Parquet on first load.
        """
        # Try parquet first
        parquet_path = self._resolve_parquet(name)
        if parquet_path:
            df = pl.read_parquet(parquet_path)
            logger.info(f"Loaded {name} from Parquet ({df.height} rows)")
            return df

        # Try external parquet
        ext_path = self._external_dir / f"{name}.parquet"
        if ext_path.exists():
            df = pl.read_parquet(ext_path)
            logger.info(f"Loaded {name} from external Parquet ({df.height} rows)")
            return df

        # Fallback: CSV
        csv_path = self._resolve_csv(name)
        if csv_path:
            df = pl.read_csv(csv_path, try_parse_dates=True)
            logger.info(f"Loaded {name} from CSV ({df.height} rows), converting to Parquet")
            # Auto-save as parquet for next time
            self.save_parquet(df, name)
            return df

        raise FileNotFoundError(
            f"Dataset '{name}' not found in parquet, external, raw, or synthetic directories. "
            f"Run `python scripts/fetch_data.py` or `python scripts/generate_data.py` first."
        )

    def load_lazy(self, name: str) -> pl.LazyFrame:
        """Load as LazyFrame for deferred computation."""
        parquet_path = self._resolve_parquet(name)
        if parquet_path:
            return pl.scan_parquet(parquet_path)

        ext_path = self._external_dir / f"{name}.parquet"
        if ext_path.exists():
            return pl.scan_parquet(ext_path)

        # Force parquet creation from CSV
        df = self.load(name)
        path = self._parquet_dir / f"{name}.parquet"
        return pl.scan_parquet(path)

    def load_commodity_prices(self) -> pl.DataFrame:
        return self.load("commodity_prices")

    def load_sales_data(self) -> pl.DataFrame:
        return self.load("sales_data")

    def load_macro_indicators(self) -> pl.DataFrame:
        return self.load("macro_indicators")

    def load_production_inventory(self) -> pl.DataFrame:
        return self.load("production_inventory")

    def load_bom_data(self) -> pl.DataFrame:
        return self.load("bom_data")

    def load_market_data(self) -> pl.DataFrame:
        """Load real-world market data (from yfinance)."""
        return self.load("market_commodities")

    def load_crypto_data(self) -> pl.DataFrame:
        """Load crypto market data (from ccxt)."""
        return self.load("crypto_prices")

    def load_fred_data(self) -> pl.DataFrame:
        """Load FRED macro data."""
        return self.load("fred_macro")

    def convert_all_csv_to_parquet(self) -> dict[str, Path]:
        """Batch convert all existing CSV files to Parquet."""
        results = {}
        for d in [self._synthetic_dir, self._raw_dir]:
            if not d.exists():
                continue
            for csv_file in d.glob("*.csv"):
                name = csv_file.stem
                if not (self._parquet_dir / f"{name}.parquet").exists():
                    df = pl.read_csv(csv_file, try_parse_dates=True)
                    path = self.save_parquet(df, name)
                    results[name] = path
        return results

    def list_datasets(self) -> dict[str, list[str]]:
        """List all available datasets by source."""
        result = {}
        for label, d in [
            ("parquet", self._parquet_dir),
            ("external", self._external_dir),
            ("raw", self._raw_dir),
            ("synthetic", self._synthetic_dir),
        ]:
            if d.exists():
                files = [f.stem for f in d.glob("*.*") if f.suffix in (".parquet", ".csv")]
                if files:
                    result[label] = sorted(files)
        return result
