"""
Forecast Bias Tracking (Layer 5 — Governance & Control)

Monitors systematic over/under-forecasting by comparing
predictions against actuals over time. Triggers alerts
when bias exceeds configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings


@dataclass
class BiasReport:
    model_name: str
    commodity: str
    mean_bias_pct: float
    median_bias_pct: float
    bias_direction: str  # "over" or "under"
    is_alert: bool
    n_observations: int
    recent_bias_trend: str  # "improving", "worsening", "stable"


class BiasTracker:
    """
    Tracks forecast bias over time.

    Bias = (Forecast - Actual) / Actual × 100
    Positive bias = over-forecasting
    Negative bias = under-forecasting
    """

    def __init__(self):
        self.settings = get_settings()
        self.threshold = self.settings["governance"]["bias_threshold_pct"]
        self.history: list[dict] = []

    def compute_bias(
        self,
        actuals: pd.Series,
        forecasts: pd.Series,
        model_name: str = "",
        commodity: str = "",
    ) -> BiasReport:
        """Compute forecast bias metrics."""
        # Avoid division by zero
        mask = actuals.abs() > 1e-10
        actual_vals = actuals[mask]
        forecast_vals = forecasts[mask]

        bias_pct = ((forecast_vals.values - actual_vals.values) / actual_vals.values) * 100

        mean_bias = float(np.mean(bias_pct))
        median_bias = float(np.median(bias_pct))
        direction = "over" if mean_bias > 0 else "under"
        is_alert = abs(mean_bias) > self.threshold

        # Trend: compare recent half vs older half
        if len(bias_pct) >= 6:
            mid = len(bias_pct) // 2
            old_bias = abs(np.mean(bias_pct[:mid]))
            new_bias = abs(np.mean(bias_pct[mid:]))
            if new_bias < old_bias * 0.8:
                trend = "improving"
            elif new_bias > old_bias * 1.2:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        report = BiasReport(
            model_name=model_name,
            commodity=commodity,
            mean_bias_pct=round(mean_bias, 2),
            median_bias_pct=round(median_bias, 2),
            bias_direction=direction,
            is_alert=is_alert,
            n_observations=len(bias_pct),
            recent_bias_trend=trend,
        )

        if is_alert:
            logger.warning(
                f"BIAS ALERT: {model_name}/{commodity} bias={mean_bias:.1f}% "
                f"(threshold={self.threshold}%), direction={direction}"
            )

        self.history.append({
            "model_name": model_name,
            "commodity": commodity,
            "mean_bias_pct": mean_bias,
            "is_alert": is_alert,
        })

        return report

    def summary_table(self) -> pd.DataFrame:
        """Return a summary of all tracked bias reports."""
        return pd.DataFrame(self.history)
