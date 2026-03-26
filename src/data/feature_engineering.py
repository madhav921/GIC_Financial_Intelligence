"""
Feature engineering for commodity and financial models.
Transforms raw time-series into ML-ready features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged values for specified columns."""
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean and std for specified columns."""
    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_ma{w}"] = df[col].rolling(w).mean()
            df[f"{col}_std{w}"] = df[col].rolling(w).std()
    return df


def add_pct_change(df: pd.DataFrame, columns: list[str], periods: list[int]) -> pd.DataFrame:
    """Add percentage change features."""
    df = df.copy()
    for col in columns:
        for p in periods:
            df[f"{col}_pctchg{p}"] = df[col].pct_change(p)
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add month, quarter, year features from a date column."""
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def prepare_commodity_features(
    commodity_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    target_commodity: str,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix for a single commodity forecast model.

    Merges commodity prices with macro indicators, adds tech features.
    Returns a DataFrame ready for train/test split.
    """
    lags = lags or [1, 3, 6, 12]
    windows = windows or [3, 6, 12]

    # Merge commodity with macro on date
    df = commodity_df[["date", target_commodity]].merge(macro_df, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # Rename target
    df = df.rename(columns={target_commodity: "target"})

    # Add features from target
    df = add_lag_features(df, ["target"], lags)
    df = add_rolling_features(df, ["target"], windows)
    df = add_pct_change(df, ["target"], [1, 3, 6])

    # Add macro lags
    macro_cols = [c for c in macro_df.columns if c != "date"]
    df = add_lag_features(df, macro_cols, [1, 3])
    df = add_pct_change(df, macro_cols, [1, 3])

    # Calendar
    df = add_calendar_features(df)

    # Drop rows with NaN from lagging
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Feature matrix for {target_commodity}: {len(df)} rows (dropped {n_before - len(df)} for lags)")

    return df


def prepare_demand_features(
    sales_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    commodity_df: pd.DataFrame,
    segment: str,
) -> pd.DataFrame:
    """Build feature matrix for demand forecasting of a given segment."""
    seg_df = sales_df[sales_df["segment"] == segment].copy()
    # Aggregate by month
    monthly = seg_df.groupby("date").agg(
        volume=("volume", "sum"),
        avg_price=("avg_price_usd", "mean"),
        avg_incentive=("incentive_pct", "mean"),
    ).reset_index()

    # Merge macro
    df = monthly.merge(macro_df, on="date", how="inner")

    # Merge average commodity index
    comm_cols = [c for c in commodity_df.columns if c != "date"]
    commodity_df = commodity_df.copy()
    commodity_df["commodity_index"] = commodity_df[comm_cols].mean(axis=1)
    df = df.merge(commodity_df[["date", "commodity_index"]], on="date", how="inner")

    df = df.sort_values("date").reset_index(drop=True)

    # Rename
    df = df.rename(columns={"volume": "target"})

    # Features
    df = add_lag_features(df, ["target", "avg_price", "commodity_index"], [1, 3, 6])
    df = add_rolling_features(df, ["target"], [3, 6])
    df = add_pct_change(df, ["target", "commodity_index"], [1, 3])
    df = add_calendar_features(df)

    df = df.dropna().reset_index(drop=True)
    logger.info(f"Demand features for {segment}: {len(df)} rows")
    return df
