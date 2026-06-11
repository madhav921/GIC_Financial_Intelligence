"""
Layer 2: Predictive Intelligence Controller

Single entry point for all ML forecasting, regime detection, and model management.
Delegates to src/models/ modules.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
from loguru import logger

from src.models.commodity_forecast import CommodityForecastModel, ForecastResult
from src.models.regime_detector import RegimeDetector
from src.config import get_settings


class IntelligenceLayerController:
    """
    Layer 2 Controller — Predictive Intelligence

    Usage:
        intel = IntelligenceLayerController()
        metrics = intel.train_all_models(commodity_df, macro_df)
        result  = intel.forecast_commodity("Copper", commodity_df, macro_df)
        index   = intel.generate_commodity_index(commodity_df)
        regime  = intel.detect_regime(price_series)
    """

    def __init__(self):
        self.settings = get_settings()
        self._model = CommodityForecastModel()
        self._regime_detector = RegimeDetector()
        self._training_metrics: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def train_all_models(
        self,
        commodity_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, dict]:
        """
        Train SARIMAX + XGBoost + cross-validation for all 12 configured commodities.
        Returns per-commodity metrics dict.
        """
        logger.info("Layer 2: Training all commodity models (SARIMAX + XGBoost, 5-fold CV)")
        self._training_metrics = self._model.train_all_commodities(commodity_df, macro_df)
        logger.info(f"Layer 2: Trained models for {len(self._training_metrics)} commodities")
        return self._training_metrics

    def forecast_commodity(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """
        Generate ensemble forecast for a single commodity.
        Uses regime-adaptive SARIMAX + XGBoost blend.
        """
        logger.info(f"Layer 2: Forecasting {commodity} (ensemble, regime-adaptive)")
        return self._model.forecast_ensemble(commodity, commodity_df, macro_df)

    def forecast_all_commodities(
        self,
        commodity_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, ForecastResult]:
        """Generate forecasts for all trained commodities."""
        logger.info("Layer 2: Generating forecasts for all commodities")
        results = {}
        commodity_cols = [c for c in commodity_df.columns if c != "date"]
        for commodity in commodity_cols:
            try:
                if commodity in self._model.sarimax_models or commodity in self._model.xgb_models:
                    results[commodity] = self.forecast_commodity(commodity, commodity_df, macro_df)
            except Exception as e:
                logger.warning(f"Layer 2: Forecast failed for {commodity} — {e}")
        return results

    def forecast_auto(
        self,
        commodity: str,
        commodity_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, ForecastResult]:
        """
        Auto-route to all applicable forecast methods (primary + futures curve + scenario).
        Returns dict with keys: 'primary', 'futures_curve', 'scenario'.
        """
        return self._model.forecast_auto(commodity, commodity_df, macro_df)

    def generate_commodity_index(self, commodity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BOM-weighted commodity index normalized to base 100.
        Formula: Index(t) = Σ(w_i × Price_i(t) / Price_i(0) × 100) / Σ(w_i)
        """
        logger.info("Layer 2: Computing BOM-weighted commodity index")
        return self._model.generate_commodity_index(commodity_df)

    def detect_regime(self, price_series: "np.ndarray") -> dict:
        """
        Detect market regime using Hurst exponent.
        Returns: {regime, hurst, confidence, ensemble_weights}
        """
        return self._regime_detector.detect(price_series)

    def get_feature_importance(self, commodity: str) -> pd.DataFrame:
        """Return XGBoost feature importance for a commodity."""
        return self._model.get_feature_importance(commodity)

    def get_cv_metrics(self) -> pd.DataFrame:
        """Return cross-validation MAPE metrics for all commodities."""
        return self._model.get_cv_metrics()

    def run_monthly_update(
        self,
        commodity_df: pd.DataFrame,
        prior_forecasts: dict[str, float],
    ) -> dict:
        """Execute 7-step monthly variance tracking and model update cycle."""
        return self._model.run_monthly_update(commodity_df, prior_forecasts)
