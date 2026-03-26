"""Shared helpers for dashboard pages."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st
from loguru import logger

# Project root
_ROOT = Path(__file__).resolve().parents[3]


@st.cache_data(ttl=3600)
def load_parquet(name: str) -> pl.DataFrame | None:
    """Load a Parquet or CSV dataset with caching."""
    for parent_dir in ["data/external", "data/parquet", "data/synthetic"]:
        parquet_path = _ROOT / parent_dir / f"{name}.parquet"
        if parquet_path.exists():
            return pl.read_parquet(parquet_path)

    csv_path = _ROOT / "data" / "synthetic" / f"{name}.csv"
    if csv_path.exists():
        return pl.read_csv(csv_path, try_parse_dates=True)

    return None


def format_currency(value: float, prefix: str = "$") -> str:
    """Format a number as currency."""
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{prefix}{value / 1e3:.1f}K"
    return f"{prefix}{value:,.2f}"


def format_pct(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:+.2f}%"


def metric_card_css():
    """Return CSS for styled metric cards."""
    return """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card h3 {
        color: #a8a8b3;
        font-size: 0.85rem;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .metric-card .value {
        color: #e0e0e0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-card .delta-positive {
        color: #00d4aa;
        font-size: 0.9rem;
    }
    .metric-card .delta-negative {
        color: #ff6b6b;
        font-size: 0.9rem;
    }
    .alert-critical {
        background: rgba(255, 59, 48, 0.15);
        border-left: 4px solid #ff3b30;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .alert-warning {
        background: rgba(255, 204, 0, 0.15);
        border-left: 4px solid #ffcc00;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .alert-info {
        background: rgba(0, 122, 255, 0.15);
        border-left: 4px solid #007aff;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    </style>
    """
