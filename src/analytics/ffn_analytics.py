"""
FFN (Financial Functions for Python) analytics module.

Wraps the ffn library to provide:
  - Performance analytics (CAGR, Sharpe, Max Drawdown, Calmar)
  - Correlation analysis across commodity baskets
  - Rolling risk metrics
  - Benchmark comparison for commodity portfolios
"""

from __future__ import annotations

import polars as pl
from loguru import logger

try:
    import ffn
    import pandas as pd
    FFN_AVAILABLE = True
except ImportError:
    FFN_AVAILABLE = False
    logger.warning("ffn not installed — financial analytics module unavailable")


def compute_performance_stats(prices_df: pl.DataFrame, date_col: str = "date") -> dict:
    """
    Compute comprehensive performance statistics for each price series.

    Args:
        prices_df: Polars DataFrame with date + numeric price columns
        date_col: Name of the date column

    Returns:
        Dict of {column_name: {metric: value}} with CAGR, Sharpe, Sortino,
        max_drawdown, calmar_ratio, volatility, etc.
    """
    if not FFN_AVAILABLE:
        logger.warning("ffn not available")
        return {}

    pdf = prices_df.to_pandas()
    pdf[date_col] = pd.to_datetime(pdf[date_col])
    pdf = pdf.set_index(date_col)

    # Drop non-numeric columns
    numeric_cols = pdf.select_dtypes(include="number").columns.tolist()
    pdf = pdf[numeric_cols]

    results = {}
    for col in numeric_cols:
        try:
            series = pdf[col].dropna()
            if len(series) < 10:
                continue

            perf = ffn.PerformanceStats(series)
            stats = perf.stats

            results[col] = {
                "total_return": float(stats.loc["total_return", col]) if "total_return" in stats.index else None,
                "cagr": float(stats.loc["cagr", col]) if "cagr" in stats.index else None,
                "daily_sharpe": float(stats.loc["daily_sharpe", col]) if "daily_sharpe" in stats.index else None,
                "daily_sortino": float(stats.loc["daily_sortino", col]) if "daily_sortino" in stats.index else None,
                "max_drawdown": float(stats.loc["max_drawdown", col]) if "max_drawdown" in stats.index else None,
                "calmar": float(stats.loc["calmar", col]) if "calmar" in stats.index else None,
                "daily_vol": float(stats.loc["daily_vol", col]) if "daily_vol" in stats.index else None,
                "monthly_vol": float(stats.loc["monthly_vol", col]) if "monthly_vol" in stats.index else None,
                "avg_drawdown": float(stats.loc["avg_drawdown", col]) if "avg_drawdown" in stats.index else None,
                "best_day": float(stats.loc["best_day", col]) if "best_day" in stats.index else None,
                "worst_day": float(stats.loc["worst_day", col]) if "worst_day" in stats.index else None,
            }
        except Exception as e:
            logger.warning(f"FFN stats failed for {col}: {e}")
            results[col] = {}

    return results


def compute_correlation_matrix(prices_df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """
    Compute return-based correlation matrix across all price series.

    Returns:
        Polars DataFrame with correlation matrix
    """
    if not FFN_AVAILABLE:
        return pl.DataFrame()

    pdf = prices_df.to_pandas()
    pdf[date_col] = pd.to_datetime(pdf[date_col])
    pdf = pdf.set_index(date_col)
    numeric_cols = pdf.select_dtypes(include="number").columns.tolist()
    pdf = pdf[numeric_cols]

    # Compute returns then correlation
    returns = pdf.pct_change().dropna()
    corr = returns.corr()

    result = corr.reset_index()
    result.columns = ["commodity"] + list(corr.columns)
    return pl.from_pandas(result)


def compute_rolling_metrics(
    prices_df: pl.DataFrame,
    window: int = 60,
    date_col: str = "date",
) -> pl.DataFrame:
    """
    Compute rolling risk metrics (volatility, Sharpe, drawdown) for each series.

    Args:
        prices_df: Polars DataFrame with price data
        window: Rolling window size in periods
        date_col: Date column name

    Returns:
        Polars DataFrame with rolling metrics
    """
    pdf = prices_df.to_pandas()
    pdf[date_col] = pd.to_datetime(pdf[date_col])
    pdf = pdf.set_index(date_col)
    numeric_cols = pdf.select_dtypes(include="number").columns.tolist()

    returns = pdf[numeric_cols].pct_change()

    result_frames = []
    for col in numeric_cols:
        rolling_vol = returns[col].rolling(window).std()
        rolling_mean = returns[col].rolling(window).mean()
        rolling_sharpe = rolling_mean / rolling_vol

        # Max drawdown window
        cumulative = (1 + returns[col]).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max

        frame = pd.DataFrame({
            "date": pdf.index,
            "commodity": col,
            "rolling_volatility": rolling_vol.values,
            "rolling_sharpe": rolling_sharpe.values,
            "rolling_drawdown": drawdown.values,
            "rolling_return": rolling_mean.values,
        })
        result_frames.append(frame)

    if not result_frames:
        return pl.DataFrame()

    combined = pd.concat(result_frames, ignore_index=True)
    return pl.from_pandas(combined)


def compute_drawdown_analysis(prices_df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """Compute drawdown series for each commodity."""
    pdf = prices_df.to_pandas()
    pdf[date_col] = pd.to_datetime(pdf[date_col])
    pdf = pdf.set_index(date_col)
    numeric_cols = pdf.select_dtypes(include="number").columns.tolist()

    result_frames = []
    for col in numeric_cols:
        series = pdf[col].dropna()
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max

        frame = pd.DataFrame({
            "date": series.index,
            "commodity": col,
            "price": series.values,
            "running_max": running_max.values,
            "drawdown_pct": drawdown.values,
        })
        result_frames.append(frame)

    if not result_frames:
        return pl.DataFrame()

    return pl.from_pandas(pd.concat(result_frames, ignore_index=True))
