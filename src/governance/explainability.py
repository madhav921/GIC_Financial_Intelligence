"""
Explainability Module (Layer 5 — Governance & Control)

Provides model interpretability through:
  - Feature importance rankings
  - SHAP-like contribution analysis (simplified)
  - Natural-language forecast explanations

Ensures AI model outputs are transparent and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ForecastExplanation:
    """Human-readable explanation for a forecast."""
    commodity: str
    forecast_value: float
    forecast_change_pct: float
    top_drivers: list[dict]  # [{feature, contribution, direction}]
    narrative: str


class ExplainabilityEngine:
    """
    Generates explanations for model predictions.

    Uses feature importance and contribution analysis to create
    human-readable explanations for executive dashboards.
    """

    def explain_commodity_forecast(
        self,
        commodity: str,
        forecast_value: float,
        current_value: float,
        feature_importance: pd.DataFrame,
        feature_values: dict[str, float] | None = None,
    ) -> ForecastExplanation:
        """Generate an explanation for a commodity price forecast."""
        change_pct = (forecast_value - current_value) / current_value * 100

        # Top driving features
        top_n = min(5, len(feature_importance))
        top_features = feature_importance.head(top_n)

        drivers = []
        for _, row in top_features.iterrows():
            feature = row["feature"]
            importance = row["importance"]

            # Determine direction from feature name heuristics
            direction = self._infer_direction(feature, feature_values)

            drivers.append({
                "feature": feature,
                "importance": round(float(importance), 4),
                "direction": direction,
            })

        # Generate narrative
        narrative = self._build_narrative(commodity, forecast_value, change_pct, drivers)

        return ForecastExplanation(
            commodity=commodity,
            forecast_value=forecast_value,
            forecast_change_pct=round(change_pct, 2),
            top_drivers=drivers,
            narrative=narrative,
        )

    def _infer_direction(self, feature: str, values: dict | None) -> str:
        """Infer whether a feature is pushing price up or down."""
        if values is None:
            return "unknown"

        val = values.get(feature, 0)
        if "pctchg" in feature:
            return "up" if val > 0 else "down"
        if "lag" in feature:
            return "neutral"
        return "up" if val > 0 else "down"

    def _build_narrative(
        self,
        commodity: str,
        forecast: float,
        change_pct: float,
        drivers: list[dict],
    ) -> str:
        """Build a natural-language explanation."""
        direction = "increase" if change_pct > 0 else "decrease"
        magnitude = "significantly" if abs(change_pct) > 10 else "moderately" if abs(change_pct) > 5 else "slightly"

        top_driver_names = [d["feature"].replace("_", " ") for d in drivers[:3]]
        driver_text = ", ".join(top_driver_names)

        narrative = (
            f"{commodity} is forecast to {magnitude} {direction} by {abs(change_pct):.1f}% "
            f"to ${forecast:,.0f}. "
            f"Key drivers: {driver_text}. "
        )

        # Add context based on feature types
        for d in drivers[:2]:
            if "oil" in d["feature"].lower():
                narrative += f"Oil price movements are a {'positive' if d['direction'] == 'up' else 'negative'} factor. "
            if "gdp" in d["feature"].lower():
                narrative += f"GDP trends point {'upward' if d['direction'] == 'up' else 'downward'}. "
            if "ma" in d["feature"] and d["direction"] == "up":
                narrative += "Recent price momentum is pushing higher. "

        return narrative.strip()

    def model_card(
        self,
        model_name: str,
        metrics: dict[str, float],
        training_period: str,
        features_used: list[str],
    ) -> str:
        """Generate a model card (documentation) for governance."""
        card = f"""
# Model Card: {model_name}

## Overview
- **Training Period**: {training_period}
- **Features**: {len(features_used)} input features
- **Last Updated**: Current

## Performance Metrics
"""
        for metric, value in metrics.items():
            card += f"- **{metric.upper()}**: {value:.4f}\n"

        card += f"""
## Features Used
"""
        for f in features_used[:20]:
            card += f"- {f}\n"
        if len(features_used) > 20:
            card += f"- ... and {len(features_used) - 20} more\n"

        card += """
## Limitations
- Model trained on synthetic data for development
- Real-world performance requires validation with actual market data
- Predictions assume stable macroeconomic regime
- Does not account for black swan events or structural breaks
"""
        return card
