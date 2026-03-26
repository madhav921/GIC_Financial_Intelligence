"""Tests for commodity forecast model."""

import numpy as np
import pandas as pd
import pytest

from src.models.commodity_forecast import (
    CommodityForecastModel,
    SARIMAXForecaster,
    XGBoostForecaster,
)


class TestSARIMAXForecaster:
    def test_fit_and_predict(self, commodity_df):
        model = SARIMAXForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 12))
        metrics = model.fit(commodity_df["Lithium"])
        assert "mae" in metrics
        assert metrics["mae"] > 0

        mean, ci_80, ci_95 = model.predict(6)
        assert len(mean) == 6
        assert ci_80.shape == (6, 2)
        assert ci_95.shape == (6, 2)
        # Upper bound should be >= lower bound
        assert all(ci_95[:, 1] >= ci_95[:, 0])


class TestXGBoostForecaster:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        })
        y = pd.Series(X["f1"] * 2 + X["f2"] + rng.normal(0, 0.1, n))

        model = XGBoostForecaster(n_estimators=50)
        metrics = model.fit(X[:80], y[:80], X[80:], y[80:])

        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["mae"] < 2  # should fit well

        preds = model.predict(X[80:])
        assert len(preds) == 20

    def test_feature_importance(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"important": rng.standard_normal(100), "noise": rng.standard_normal(100)})
        y = pd.Series(X["important"] * 5 + rng.normal(0, 0.1, 100))

        model = XGBoostForecaster(n_estimators=50)
        model.fit(X, y)
        fi = model.feature_importance()
        assert fi.iloc[0]["feature"] == "important"


class TestCommodityForecastModel:
    def test_train_xgboost(self, commodity_df, macro_df):
        cfm = CommodityForecastModel()
        metrics = cfm.train_xgboost("Lithium", commodity_df, macro_df, test_size=6)
        assert "mae" in metrics
        assert "mape" in metrics

    def test_commodity_index(self, commodity_df):
        cfm = CommodityForecastModel()
        index_df = cfm.generate_commodity_index(commodity_df)
        assert "commodity_index" in index_df.columns
        assert len(index_df) == len(commodity_df)
        # Index should start near 100
        assert 90 < index_df["commodity_index"].iloc[0] < 110

    def test_cross_validate(self, commodity_df, macro_df):
        cfm = CommodityForecastModel()
        cv = cfm.cross_validate("Steel", commodity_df, macro_df, n_splits=3)
        assert "cv_mae_mean" in cv
        assert cv["cv_mae_mean"] > 0
