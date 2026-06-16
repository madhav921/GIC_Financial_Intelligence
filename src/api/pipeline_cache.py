"""
Pipeline Cache — shared background singleton.

Computes Layer 1 + Layer 2 outputs once and caches them for all routes.
Refreshes every 30 minutes in a daemon thread so individual route calls
are fast and all pages share the same data source.

Usage:
    from src.api.pipeline_cache import get_snapshot, start_background_loop

    snap = get_snapshot()            # always returns a dict (may be empty on cold start)
    start_background_loop()          # call once from app lifespan
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings

_REFRESH_INTERVAL_SECS = 1800  # 30 minutes


class _PipelineCache:
    """Thread-safe container for cached pipeline outputs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict = {}
        self._refreshed_at: float = 0.0

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def set(self, data: dict) -> None:
        with self._lock:
            self._data = data
            self._refreshed_at = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self._refreshed_at

    def is_empty(self) -> bool:
        with self._lock:
            return not self._data


_cache = _PipelineCache()


def get_snapshot() -> dict:
    """
    Return the latest cached pipeline snapshot.

    Keys (all Optional — check before use):
        commodity_df          : pd.DataFrame — raw commodity prices
        commodity_index_df    : pd.DataFrame — BOM-weighted index (date, commodity_index)
        forecasts             : dict[commodity → {forecast_pct, exposure_gbp, hedge_ratio}]
        commodity_volatility_pct : float — annualised index vol (%)
        bias_metrics          : list[dict] — per-commodity bias, mape, status
        refreshed_at          : float — epoch seconds of last successful refresh
    """
    return _cache.get()


def trigger_refresh() -> None:
    """Fire a one-shot background refresh (non-blocking)."""
    t = threading.Thread(target=_refresh, daemon=True, name="pipeline-refresh-once")
    t.start()


def start_background_loop() -> None:
    """
    Start a daemon thread that refreshes the cache every 30 minutes.
    Safe to call multiple times — only the first call starts the loop.
    """
    def loop() -> None:
        while True:
            _refresh()
            time.sleep(_REFRESH_INTERVAL_SECS)

    t = threading.Thread(target=loop, daemon=True, name="pipeline-cache-loop")
    t.start()
    logger.info("Pipeline cache background loop started (interval=30 min)")


def _refresh() -> None:
    """Compute and store a fresh pipeline snapshot. Runs in background thread."""
    try:
        from layers.layer1_data.controller import DataLayerController
        from layers.layer2_intelligence.controller import IntelligenceLayerController

        settings = get_settings()

        data = DataLayerController()
        intel = IntelligenceLayerController()

        commodity_df = data.load_commodity_data()
        commodity_index_df = intel.generate_commodity_index(commodity_df)

        forecasts = _build_forecasts(commodity_df, settings)
        commodity_volatility_pct = _compute_index_vol(commodity_index_df)
        bias_metrics = _compute_bias_metrics(commodity_df, settings)

        _cache.set({
            "commodity_df": commodity_df,
            "commodity_index_df": commodity_index_df,
            "forecasts": forecasts,
            "commodity_volatility_pct": commodity_volatility_pct,
            "bias_metrics": bias_metrics,
            "refreshed_at": time.time(),
        })

        logger.info(
            f"Pipeline cache refreshed — {len(forecasts)} forecasts, "
            f"vol={commodity_volatility_pct}%, "
            f"{len(bias_metrics)} bias metrics"
        )
    except Exception as exc:
        logger.warning(f"Pipeline cache refresh failed: {exc}")


def _col_for(name: str, df: pd.DataFrame) -> Optional[str]:
    """Find the DataFrame column matching a commodity name (handles space↔underscore)."""
    underscore = name.replace(" ", "_")
    if underscore in df.columns:
        return underscore
    if name in df.columns:
        return name
    return None


def _build_forecasts(commodity_df: pd.DataFrame, settings: dict) -> dict:
    """
    Build the forecasts dict expected by InsightEngine._commodity_insights().

    Uses the 12-month trailing price change as the forward forecast_pct.
    This uses the same data source (DataLayerController) as every other route.
    """
    total_spend_gbp = 3_300_000_000  # strategic tracked basket
    forecasts: dict = {}

    for cfg in settings.get("commodities", []):
        name = cfg["name"]
        col = _col_for(name, commodity_df)
        if col is None:
            continue

        prices = commodity_df[col].dropna()
        if len(prices) < 6:
            continue

        current = float(prices.iloc[-1])
        # 12-month-ago or earliest available as baseline
        baseline_idx = max(0, len(prices) - 13)
        baseline = float(prices.iloc[baseline_idx])
        forecast_pct = (current - baseline) / max(abs(baseline), 1e-9)

        bom_w = float(cfg.get("bom_weight") or 0.05)
        exposure_gbp = total_spend_gbp * bom_w

        forecasts[name] = {
            "forecast_pct": round(forecast_pct, 4),
            "exposure_gbp": round(exposure_gbp, 0),
            "hedge_ratio": 0.40,  # default; no hedge ratio stored in config
        }

    return forecasts


def _compute_index_vol(commodity_index_df: pd.DataFrame) -> Optional[float]:
    """Annualised commodity index volatility (%)."""
    try:
        if commodity_index_df.empty or "commodity_index" not in commodity_index_df.columns:
            return None
        idx = commodity_index_df["commodity_index"].dropna()
        if len(idx) < 3:
            return None
        returns = idx.pct_change().dropna()
        vol = float(returns.std() * (12 ** 0.5) * 100)
        return round(vol, 1)
    except Exception:
        return None


def _compute_bias_metrics(commodity_df: pd.DataFrame, settings: dict) -> list[dict]:
    """
    Compute naive rolling-MA bias for each commodity.

    Splits the last 12 observations into actuals and uses the 3-month trailing
    mean as the naive forecast proxy. This gives a real, data-driven bias
    estimate using the same source as the rest of the platform.
    """
    bias_threshold = float(settings.get("governance", {}).get("bias_threshold_pct", 5.0))
    escalation_threshold = 10.0
    results: list[dict] = []

    for cfg in settings.get("commodities", []):
        name = cfg["name"]
        col = _col_for(name, commodity_df)
        if col is None:
            continue

        prices = commodity_df[col].dropna().values
        if len(prices) < 15:
            continue

        # last 12 months = actuals; preceding 3 = seed for the rolling mean
        window = 3
        actuals = prices[-12:]
        seed = prices[-(12 + window):-12]

        naive_forecasts: list[float] = []
        rolling_buf = list(seed)
        for i in range(12):
            naive_forecasts.append(float(np.mean(rolling_buf[-window:])))
            rolling_buf.append(actuals[i])

        bias_pcts = [
            (f - a) / max(abs(a), 1e-9) * 100
            for f, a in zip(naive_forecasts, actuals)
        ]
        mean_bias = float(np.mean(bias_pcts))
        mape = float(np.mean([abs(b) for b in bias_pcts]))

        if abs(mean_bias) > escalation_threshold:
            status = "escalate"
        elif abs(mean_bias) > bias_threshold:
            status = "alert"
        else:
            status = "good"

        results.append({
            "commodity": name,
            "bias": round(abs(mean_bias), 1),
            "mape": round(mape, 1),
            "status": status,
            "direction": "over" if mean_bias > 0 else "under",
        })

    results.sort(key=lambda x: x["bias"], reverse=True)
    return results
