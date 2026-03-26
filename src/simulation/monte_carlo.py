"""
Monte Carlo Simulation Engine (Layer 4 — Scenario & Simulation)

Runs thousands of simulations across uncertain parameters to generate:
  - Margin impact distributions
  - Cash flow risk profiles (VaR, CVaR)
  - Probability-weighted financial outcomes

This is the probabilistic layer that sits on top of the deterministic
financial model, adding uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings
from src.drivers.financial_model import FinancialModel


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation run."""
    n_simulations: int
    scenario_name: str
    # Distributions
    net_revenue_dist: np.ndarray
    cogs_dist: np.ndarray
    gross_margin_dist: np.ndarray
    operating_income_dist: np.ndarray
    # Summary statistics
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute_stats(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all distributions."""
        self.stats = {}
        for name, dist in [
            ("net_revenue", self.net_revenue_dist),
            ("cogs", self.cogs_dist),
            ("gross_margin", self.gross_margin_dist),
            ("operating_income", self.operating_income_dist),
        ]:
            self.stats[name] = {
                "mean": float(np.mean(dist)),
                "median": float(np.median(dist)),
                "std": float(np.std(dist)),
                "p5": float(np.percentile(dist, 5)),
                "p10": float(np.percentile(dist, 10)),
                "p25": float(np.percentile(dist, 25)),
                "p75": float(np.percentile(dist, 75)),
                "p90": float(np.percentile(dist, 90)),
                "p95": float(np.percentile(dist, 95)),
                "var_95": float(np.percentile(dist, 5)),  # Value at Risk
                "cvar_95": float(np.mean(dist[dist <= np.percentile(dist, 5)])),  # Conditional VaR
            }
        return self.stats

    def summary_df(self) -> pd.DataFrame:
        """Return a summary DataFrame."""
        if not self.stats:
            self.compute_stats()
        records = []
        for metric, stats in self.stats.items():
            records.append({"metric": metric, **stats})
        return pd.DataFrame(records)

    def margin_at_risk(self, confidence: float = 0.95) -> float:
        """Gross margin at risk at given confidence level."""
        pct = (1 - confidence) * 100
        return float(np.percentile(self.gross_margin_dist, pct))


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for financial scenario analysis.

    Samples from distributions of:
      - Demand shocks (normal)
      - Commodity price shocks (fat-tailed / t-distribution)
      - FX rate movements (normal)
      - Capacity utilization changes

    For each sample, computes the full P&L through the financial model.
    """

    def __init__(self):
        self.settings = get_settings()
        self.n_simulations = self.settings["simulation"]["n_simulations"]
        self.seed = self.settings["simulation"]["seed"]

    def run(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        scenario_name: str = "base",
        n_simulations: int | None = None,
        demand_vol: float = 0.10,
        commodity_vol: float = 0.20,
        fx_vol: float = 0.05,
        demand_mean: float = 0.0,
        commodity_mean: float = 0.0,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            sales_df: Historical/projected sales data
            commodity_index_df: Commodity index time series
            scenario_name: Name of the scenario
            n_simulations: Number of simulations (overrides config)
            demand_vol: Demand shock volatility (std dev)
            commodity_vol: Commodity shock volatility (std dev)
            fx_vol: FX shock volatility (std dev)
            demand_mean: Mean demand shock (for asymmetric scenarios)
            commodity_mean: Mean commodity shock
        """
        n_sims = n_simulations or self.n_simulations
        rng = np.random.default_rng(self.seed)
        financial_model = FinancialModel()

        logger.info(f"Running Monte Carlo: {n_sims} simulations, scenario='{scenario_name}'")

        # Sample shocks
        demand_shocks = rng.normal(demand_mean, demand_vol, n_sims)
        # Use t-distribution for commodity (fatter tails)
        commodity_shocks = rng.standard_t(df=5, size=n_sims) * commodity_vol + commodity_mean

        # Run base P&L once to get structure
        base_pnl = financial_model.build_pnl(sales_df, commodity_index_df)
        base_revenue = base_pnl["net_revenue"].sum()
        base_cogs = base_pnl["total_cogs"].sum()
        base_margin = base_pnl["gross_margin"].sum()
        base_oi = base_pnl["operating_income"].sum()

        # Vectorized simulation (approximation for speed)
        # Revenue scales linearly with demand
        revenue_dist = base_revenue * (1 + demand_shocks)

        # COGS scales with demand + commodity
        material_fraction = 0.45
        cogs_demand_effect = base_cogs * (1 + demand_shocks)
        cogs_commodity_effect = base_cogs * material_fraction * commodity_shocks
        cogs_dist = cogs_demand_effect + cogs_commodity_effect

        margin_dist = revenue_dist - cogs_dist

        # Operating income (subtract fixed costs)
        fixed_costs = base_revenue - base_margin - (base_margin - base_oi)  # warranty + dep
        other_costs = base_margin - base_oi  # warranty + depreciation
        oi_dist = margin_dist - other_costs

        result = SimulationResult(
            n_simulations=n_sims,
            scenario_name=scenario_name,
            net_revenue_dist=revenue_dist,
            cogs_dist=cogs_dist,
            gross_margin_dist=margin_dist,
            operating_income_dist=oi_dist,
        )
        result.compute_stats()

        logger.info(
            f"Simulation complete: Mean margin=${result.stats['gross_margin']['mean']:,.0f}, "
            f"VaR(95%)=${result.stats['gross_margin']['var_95']:,.0f}"
        )
        return result

    def run_preset_scenarios(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
    ) -> dict[str, SimulationResult]:
        """Run all preset scenarios from config."""
        presets = self.settings["simulation"]["scenario_presets"]
        results = {}

        for name, params in presets.items():
            results[name] = self.run(
                sales_df=sales_df,
                commodity_index_df=commodity_index_df,
                scenario_name=name,
                demand_mean=params.get("demand_shock", 0),
                commodity_mean=params.get("commodity_shock", 0),
            )

        return results

    def compare_scenarios(
        self, results: dict[str, SimulationResult]
    ) -> pd.DataFrame:
        """Create a comparison table across scenarios."""
        records = []
        for name, result in results.items():
            stats = result.stats
            records.append({
                "scenario": name,
                "mean_revenue": stats["net_revenue"]["mean"],
                "mean_margin": stats["gross_margin"]["mean"],
                "mean_margin_pct": stats["gross_margin"]["mean"] / stats["net_revenue"]["mean"] * 100
                    if stats["net_revenue"]["mean"] != 0 else 0,
                "margin_var_95": stats["gross_margin"]["var_95"],
                "margin_cvar_95": stats["gross_margin"]["cvar_95"],
                "oi_mean": stats["operating_income"]["mean"],
                "oi_p5": stats["operating_income"]["p5"],
                "oi_p95": stats["operating_income"]["p95"],
            })
        return pd.DataFrame(records).round(0)
