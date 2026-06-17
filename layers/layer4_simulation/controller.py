"""
Layer 4: Simulation & Risk Controller

Single entry point for Monte Carlo simulation, scenario analysis, and risk quantification.
Delegates to src/simulation/ and src/models/ modules.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger

from src.simulation.monte_carlo import MonteCarloEngine, SimulationResult
from src.simulation.scenario_engine import ScenarioEngine
from src.models.hedge_optimizer import HedgeOptimizer
from src.models.quantile_forecast import QuantileForecaster


class SimulationLayerController:
    """
    Layer 4 Controller — Simulation & Risk

    Usage:
        sim = SimulationLayerController()
        result   = sim.run_monte_carlo(sales_df, commodity_index_df, n_sims=10000)
        scenarios= sim.run_all_scenarios(sales_df, commodity_index_df)
        hedge    = sim.optimize_hedge("Copper", forecast_result)
        risk     = sim.decompose_risk(sales_df, commodity_index_df)
        fan      = sim.run_monthly_fan(base_pnl)
    """

    def __init__(self):
        self._mc_engine = MonteCarloEngine()
        self._scenario_engine = ScenarioEngine()
        self._hedge_optimizer = HedgeOptimizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_monte_carlo(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        scenario_name: str = "base",
        n_simulations: int = 10_000,
        demand_shock: float = 0.0,
        commodity_shock: float = 0.0,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.
        Demand uses Normal distribution; commodity uses fat-tailed t(df=5).
        Returns distributions + VaR(95%) + CVaR(95%).
        """
        logger.info(f"Layer 4: Monte Carlo — {n_simulations:,} sims, scenario='{scenario_name}'")
        return self._mc_engine.run(
            sales_df=sales_df,
            commodity_index_df=commodity_index_df,
            scenario_name=scenario_name,
            n_simulations=n_simulations,
            demand_mean=demand_shock,
            commodity_mean=commodity_shock,
        )

    def run_all_scenarios(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
    ) -> dict[str, SimulationResult]:
        """Run all 7 preset scenarios. Returns comparison dict."""
        logger.info("Layer 4: Running all 7 preset scenarios")
        return self._mc_engine.run_preset_scenarios(sales_df, commodity_index_df)

    def compare_scenarios(
        self, results: dict[str, SimulationResult]
    ) -> pd.DataFrame:
        """Build scenario comparison table (revenue, margin, VaR, CVaR)."""
        return self._mc_engine.compare_scenarios(results)

    def run_monthly_fan(
        self,
        base_pnl: pd.DataFrame,
        n_simulations: int = 2_000,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo per month to generate fan chart data.
        Returns: date, mean_oi, p5_oi, p10_oi, p25_oi, p75_oi, p90_oi, p95_oi
        """
        logger.info("Layer 4: Building monthly fan chart")
        return self._mc_engine.run_monthly_fan(base_pnl, n_simulations=n_simulations)

    def decompose_risk(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        n_simulations: int = 3_000,
    ) -> dict:
        """
        Decompose P&L variance by source.
        Returns: {commodity_pct, demand_pct, fx_pct, var_95, cvar_95}
        """
        logger.info("Layer 4: Decomposing variance by risk source")
        return self._mc_engine.decompose_variance(sales_df, commodity_index_df, n_simulations)

    # ── Quantile-Regression VaR/CVaR (G11) ───────────────────────────────────

    def quantile_var_forecast(
        self,
        commodity_index_df: pd.DataFrame,
        horizon: int = 12,
        quantiles: tuple = (0.05, 0.25, 0.50, 0.75, 0.95),
    ) -> dict:
        """
        Fit a gradient-boosted quantile regression on the commodity index history
        and produce asymmetric VaR/CVaR bounds for the forward horizon.

        Returns:
            {quantile_bands, var_5pct, cvar_5pct, var_95pct, backend, n_train}
        """
        logger.info(f"Layer 4 [G11]: Quantile regression VaR (horizon={horizon}m)")

        if "commodity_index" not in commodity_index_df.columns:
            # Fall back to first numeric column.
            num_cols = commodity_index_df.select_dtypes("number").columns.tolist()
            if not num_cols:
                return {"error": "No numeric column in commodity_index_df"}
            col = num_cols[0]
        else:
            col = "commodity_index"

        series = commodity_index_df[col].dropna().values
        n = len(series)
        if n < 24:
            return {"error": f"Series too short for quantile VaR (n={n})"}

        # Build lag features (lags 1, 3, 6) for the quantile regressor.
        import numpy as np
        lags = [1, 3, 6]
        max_lag = max(lags)
        X_rows, y_rows = [], []
        for i in range(max_lag, n):
            X_rows.append([series[i - l] for l in lags])
            y_rows.append(series[i])
        X = np.array(X_rows)
        y = np.array(y_rows)

        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train = y[:split]

        qf = QuantileForecaster(
            quantiles=tuple(sorted(quantiles)),
            n_estimators=200,
            max_depth=3,
        )
        try:
            qf.fit(X_train, y_train)
        except Exception as exc:
            return {"error": f"Quantile fit failed: {exc}"}

        # Predict on the test/recent window for current VaR.
        X_recent = X_test if len(X_test) >= 1 else X[-min(6, len(X)):]
        bands = qf.predict(X_recent)

        # Summarise: use the mean of the last-horizon predictions.
        h = min(horizon, len(X_recent))
        summary: dict = {"quantile_bands": {}, "backend": qf.backend, "n_train": split}
        for q, preds in bands.items():
            summary["quantile_bands"][str(q)] = float(np.mean(preds[-h:]))

        q_keys = sorted(bands.keys())
        lo_key, hi_key = q_keys[0], q_keys[-1]
        mid = float(np.mean(bands[0.50][-h:])) if 0.50 in bands else float(np.mean(series[-h:]))
        lo_vals = bands[lo_key][-h:]
        hi_vals = bands[hi_key][-h:]

        summary["var_5pct"] = float(np.mean(lo_vals))
        summary["cvar_5pct"] = float(np.mean(lo_vals[lo_vals <= np.percentile(lo_vals, 25)])) if len(lo_vals) > 4 else summary["var_5pct"]
        summary["var_95pct"] = float(np.mean(hi_vals))
        summary["median_forecast"] = mid

        logger.info(
            f"Layer 4 [G11]: Quantile VaR — 5th={summary['var_5pct']:.2f}, "
            f"median={mid:.2f}, 95th={summary['var_95pct']:.2f}"
        )
        return summary

    def optimize_hedge(
        self,
        commodity: str,
        exposure_units: float,
        spot_price: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> dict:
        """
        Compute portfolio-theory optimal hedge ratio.
        Returns: {optimal_ratio, expected_savings, var_reduction}
        """
        logger.info(f"Layer 4: Optimizing hedge for {commodity}")
        result = self._hedge_optimizer.optimize(
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            futures_price=spot_price,
            exposure_units=exposure_units,
        )
        result["commodity"] = commodity
        return result
