"""
Demand Forecast Model (Layer 3 — AI Forecast Layer)

Forecasts vehicle sales volume by segment using:
  - Historical sales trends
  - Price & incentive data
  - Macroeconomic indicators
  - Commodity price index (cost pressure signals)

Outputs Volume(t) which feeds into:
  - Revenue Drivers (Layer 2)
  - Financial Driver Model: Revenue = Volume × Net Price
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import get_settings
from src.data.feature_engineering import prepare_demand_features
from src.models.model_registry import ModelRegistry


@dataclass
class DemandForecastResult:
    segment: str
    dates: list[str]
    volume_forecast: list[int]
    lower_bound: list[int]
    upper_bound: list[int]
    metrics: dict[str, float] = field(default_factory=dict)


class DemandForecastModel:
    """
    Demand forecasting for vehicle segments.
    Uses XGBoost with time-series features, macro indicators,
    and commodity index as cost-pressure signal.
    """

    def __init__(self):
        self.settings = get_settings()
        self.horizon = self.settings["forecast"]["horizon_months"]
        self.registry = ModelRegistry()
        self.models: dict[str, XGBRegressor] = {}
        self.feature_cols: dict[str, list[str]] = {}
        self.residual_std: dict[str, float] = {}

    def train(
        self,
        segment: str,
        sales_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity_df: pd.DataFrame,
        test_size: int = 12,
    ) -> dict[str, float]:
        """Train demand model for a specific vehicle segment."""
        feat_df = prepare_demand_features(sales_df, macro_df, commodity_df, segment)

        feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]
        X = feat_df[feature_cols]
        y = feat_df["target"]

        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        residuals = y_test.values - preds
        self.residual_std[segment] = float(np.std(residuals))

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "mape": float(mean_absolute_percentage_error(y_test, preds) * 100),
        }

        self.models[segment] = model
        self.feature_cols[segment] = feature_cols

        self.registry.save_model(
            model,
            f"demand_{segment.lower()}",
            metrics=metrics,
            features=feature_cols,
        )

        logger.info(f"Demand model for {segment}: MAE={metrics['mae']:.0f}, MAPE={metrics['mape']:.1f}%")
        return metrics

    def train_all_segments(
        self,
        sales_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Train demand models for all configured segments."""
        results = {}
        for seg in self.settings["vehicle_segments"]:
            segment = seg["segment"]
            try:
                results[segment] = self.train(segment, sales_df, macro_df, commodity_df)
            except Exception as e:
                logger.error(f"Failed to train demand model for {segment}: {e}")
                results[segment] = {"error": str(e)}
        return results

    def feature_importance(self, segment: str) -> pd.DataFrame:
        model = self.models[segment]
        return pd.DataFrame({
            "feature": self.feature_cols[segment],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
