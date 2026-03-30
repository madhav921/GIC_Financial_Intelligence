"""
Data loader: unified interface for loading data from various sources.
Priority: data/raw/ (real-world) → data/synthetic/ (fallback).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_project_root, get_settings


class DataLoader:
    """Load data from raw (real-world) or synthetic files."""

    def __init__(self):
        self.settings = get_settings()
        self.root = get_project_root()
        self._synthetic_dir = self.root / "data" / "synthetic"
        self._raw_dir = self.root / "data" / "raw"
        self._processed_dir = self.root / "data" / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._source_map: dict[str, str] = {}

    def _resolve_path(self, name: str) -> Path:
        """Check raw first, fall back to synthetic."""
        raw_path = self._raw_dir / f"{name}.csv"
        synth_path = self._synthetic_dir / f"{name}.csv"
        if raw_path.exists():
            self._source_map[name] = "real"
            logger.info(f"Loading from raw (real-world): {raw_path}")
            return raw_path
        if synth_path.exists():
            self._source_map[name] = "synthetic"
            logger.info(f"Loading from synthetic: {synth_path}")
            return synth_path
        raise FileNotFoundError(
            f"Dataset '{name}' not found. "
            f"Run `python scripts/fetch_data.py` (real) or `python scripts/generate_data.py` (synthetic)."
        )

    def get_data_source(self, name: str) -> str:
        """Return 'real' or 'synthetic' for a previously loaded dataset."""
        return self._source_map.get(name, "unknown")

    def load_commodity_prices(self) -> pd.DataFrame:
        path = self._resolve_path("commodity_prices")
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def load_sales_data(self) -> pd.DataFrame:
        path = self._resolve_path("sales_data")
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def load_macro_indicators(self) -> pd.DataFrame:
        path = self._resolve_path("macro_indicators")
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def load_production_inventory(self) -> pd.DataFrame:
        path = self._resolve_path("production_inventory")
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def load_bom_data(self) -> pd.DataFrame:
        path = self._resolve_path("bom_data")
        return pd.read_csv(path)

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            "commodity_prices": self.load_commodity_prices(),
            "sales_data": self.load_sales_data(),
            "macro_indicators": self.load_macro_indicators(),
            "production_inventory": self.load_production_inventory(),
            "bom_data": self.load_bom_data(),
        }
