"""
Actionable-Intelligence layer.

Turns the platform's descriptive analytics (forecasts, P&L, Monte Carlo) into
prioritised, prescriptive, £-quantified insights, a plan→actual variance bridge,
anomaly flags, an early-warning score and concrete recommendations.
"""

from __future__ import annotations

from src.insights.anomaly_detector import AnomalyDetector
from src.insights.early_warning import EarlyWarningSystem
from src.insights.insight_engine import InsightEngine
from src.insights.recommendation_engine import RecommendationEngine
from src.insights.schemas import InsightCard, VarianceBridge
from src.insights.variance_bridge import VarianceBridgeAnalyzer

__all__ = [
    "AnomalyDetector",
    "EarlyWarningSystem",
    "InsightEngine",
    "InsightCard",
    "RecommendationEngine",
    "VarianceBridge",
    "VarianceBridgeAnalyzer",
]
