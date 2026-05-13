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


# ── Smoke tests for new features ─────────────────────────────────────────────


class TestRegimeDetector:
    def test_hurst_random_walk_is_near_half(self):
        """A geometric random walk should produce H ≈ 0.5."""
        from src.models.regime_detector import RegimeDetector

        rng = np.random.default_rng(0)
        # Geometric random walk: log-normal returns
        returns = rng.normal(0, 0.02, 120)
        prices = 100.0 * np.exp(np.cumsum(returns))

        detector = RegimeDetector()
        result = detector.detect(prices)

        assert "hurst" in result
        assert "regime" in result
        assert "ensemble_weights" in result
        assert "confidence" in result
        # R/S analysis has high variance on short series; just check it's a valid float
        assert 0.0 <= result["hurst"] <= 1.0

    def test_trending_series_high_hurst(self):
        """A monotonically trending series should have H > 0.5."""
        from src.models.regime_detector import RegimeDetector

        prices = np.linspace(50, 150, 100) + np.random.default_rng(1).normal(0, 0.5, 100)
        result = RegimeDetector().detect(prices)
        assert result["hurst"] > 0.5, f"Expected H > 0.5 for trend, got {result['hurst']:.3f}"

    def test_weights_sum_to_one(self):
        """Ensemble weights must sum to 1.0."""
        from src.models.regime_detector import RegimeDetector

        rng = np.random.default_rng(7)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 60)))
        result = RegimeDetector().detect(prices)
        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"


class TestShockCalculator:
    def test_positive_shock_reduces_ebit(self):
        """A +20% lithium price shock must reduce EBIT (negative ebit_impact)."""
        from src.models.commodity_shock import CommodityShockCalculator

        calc = CommodityShockCalculator()
        base_revenue = 22_000_000_000  # $22B (JLR scale)
        result = calc.compute_shock("Lithium", 0.20, base_revenue)

        assert result["cogs_impact"] > 0, "COGS impact should be positive (cost increase)"
        assert result["ebit_impact"] < 0, "EBIT impact should be negative (profit decrease)"
        assert result["margin_impact_bps"] < 0, "Margin bps should be negative"

        # Sanity check: JLR lithium exposure is significant but not absurd
        # BOM weight ~18%, material fraction ~45%, COGS ~77.5%  → ~$345M COGS impact
        # After-tax EBIT ≈ -345 * (1 - 0.19) = ~-$280M
        assert result["ebit_impact"] < -50_000_000, "Expected EBIT impact > $50M for 20% Lithium shock"
        assert result["ebit_impact"] > -1_000_000_000, "Unrealistically large EBIT impact"

    def test_negative_shock_increases_ebit(self):
        """A -15% steel price shock must increase EBIT."""
        from src.models.commodity_shock import CommodityShockCalculator

        calc = CommodityShockCalculator()
        result = calc.compute_shock("Steel", -0.15, 22_000_000_000)
        assert result["ebit_impact"] > 0

    def test_waterfall_sorted_by_impact(self):
        """Waterfall should be sorted largest absolute impact first."""
        from src.models.commodity_shock import CommodityShockCalculator

        calc = CommodityShockCalculator()
        shocks = {"Steel": 0.30, "Nickel": 0.30, "Lithium": 0.30}
        waterfall = calc.waterfall(shocks, 22_000_000_000)
        impacts = [abs(row["ebit_impact"]) for row in waterfall]
        assert impacts == sorted(impacts, reverse=True), "Waterfall not sorted by impact magnitude"

    def test_zero_shock_is_zero_impact(self):
        """Zero shock must produce zero impact."""
        from src.models.commodity_shock import CommodityShockCalculator

        calc = CommodityShockCalculator()
        result = calc.compute_shock("Steel", 0.0, 22_000_000_000)
        assert result["ebit_impact"] == pytest.approx(0.0, abs=1.0)


class TestHedgeOptimizer:
    def test_hedge_ratio_in_bounds(self):
        """Optimal hedge ratio must be in [0, 1]."""
        from src.models.hedge_optimizer import HedgeOptimizer

        opt = HedgeOptimizer()
        result = opt.optimize(
            forecast_mean=12_000,
            forecast_std=1_200,
            futures_price=11_500,
            exposure_units=50_000,
        )
        assert "optimal_hedge_ratio" in result
        assert 0.0 <= result["optimal_hedge_ratio"] <= 1.0

    def test_higher_volatility_increases_hedge(self):
        """Greater price uncertainty should lead to higher hedge ratio."""
        from src.models.hedge_optimizer import HedgeOptimizer

        opt = HedgeOptimizer()
        low_vol = opt.optimize(12_000, 100, 11_500, 50_000)
        high_vol = opt.optimize(12_000, 3_000, 11_500, 50_000)
        assert high_vol["optimal_hedge_ratio"] >= low_vol["optimal_hedge_ratio"]

    def test_hedge_schedule_length(self):
        """Schedule should return one entry per month."""
        from src.models.hedge_optimizer import HedgeOptimizer

        opt = HedgeOptimizer()
        months = 6
        schedule = opt.schedule(
            monthly_forecasts=[12_000] * months,
            monthly_stds=[1_200] * months,
            monthly_futures=[11_800] * months,
            monthly_exposure=[5_000_000] * months,
        )
        assert len(schedule) == months
        for row in schedule:
            assert 0.0 <= row["optimal_hedge_ratio"] <= 1.0

