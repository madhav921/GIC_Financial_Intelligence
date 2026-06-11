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
