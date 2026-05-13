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
        material_fraction = self.settings["financial"]["material_cogs_fraction"]
        cogs_demand_effect = base_cogs * (1 + demand_shocks)
        cogs_commodity_effect = base_cogs * material_fraction * commodity_shocks
        cogs_dist = cogs_demand_effect + cogs_commodity_effect

        margin_dist = revenue_dist - cogs_dist

        # Operating income (subtract warranty + depreciation)
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

    # ── Fan chart: monthly time-series with uncertainty bands ─────────────────

    def run_monthly_fan(
        self,
        base_pnl: pd.DataFrame,
        n_simulations: int = 2000,
        demand_vol: float = 0.10,
        commodity_vol: float = 0.20,
        fx_vol: float = 0.05,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo across each month in base_pnl and return percentile bands.

        Used by the Executive Summary fan chart.

        Args:
            base_pnl:      Monthly P&L DataFrame from FinancialModel.build_pnl()
            n_simulations: Simulations per month
            demand_vol:    Monthly demand shock std dev
            commodity_vol: Monthly commodity shock std dev (t-dist, fat tails)
            fx_vol:        Monthly FX shock std dev

        Returns:
            DataFrame with columns: date, mean_oi, p5_oi, p10_oi, p25_oi,
                                    p75_oi, p90_oi, p95_oi
            Plus summary scalars accessible as attributes:
                annual_mean_oi, var_95, cvar_95
        """
        rng = np.random.default_rng(self.seed)
        material_fraction = self.settings["financial"]["material_cogs_fraction"]

        rows = []
        for _, month_row in base_pnl.iterrows():
            base_rev = float(month_row["net_revenue"])
            base_cogs = float(month_row["total_cogs"])
            base_gm = float(month_row["gross_margin"])
            other_costs = base_gm - float(month_row["operating_income"])

            demand_shocks = rng.normal(0, demand_vol, n_simulations)
            comm_shocks = rng.standard_t(df=5, size=n_simulations) * commodity_vol
            fx_shocks = rng.normal(0, fx_vol, n_simulations)

            rev_sim = base_rev * (1 + demand_shocks) * (1 + fx_shocks)
            cogs_sim = (
                base_cogs * (1 + demand_shocks)
                + base_cogs * material_fraction * comm_shocks
            )
            oi_sim = rev_sim - cogs_sim - other_costs

            rows.append({
                "date": month_row["date"],
                "mean_oi": float(np.mean(oi_sim)),
                "p5_oi": float(np.percentile(oi_sim, 5)),
                "p10_oi": float(np.percentile(oi_sim, 10)),
                "p25_oi": float(np.percentile(oi_sim, 25)),
                "p75_oi": float(np.percentile(oi_sim, 75)),
                "p90_oi": float(np.percentile(oi_sim, 90)),
                "p95_oi": float(np.percentile(oi_sim, 95)),
            })

        fan_df = pd.DataFrame(rows)
        logger.info(
            f"Monthly fan chart: {len(fan_df)} months, "
            f"mean annual OI = {fan_df['mean_oi'].sum():,.0f}"
        )
        return fan_df

    # ── Variance decomposition ────────────────────────────────────────────────

    def decompose_variance(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        n_simulations: int = 3000,
        demand_vol: float = 0.10,
        commodity_vol: float = 0.20,
        fx_vol: float = 0.05,
    ) -> dict:
        """
        Decompose P&L uncertainty by source: commodity, demand, FX.

        Runs 3 partial simulations (holding 2 sources fixed each time)
        then attributes variance proportionally.

        Returns:
            dict with keys: commodity_pct, demand_pct, fx_pct
            (percentages that sum to 100)
        """
        rng = np.random.default_rng(self.seed)
        financial_model = FinancialModel()
        base_pnl = financial_model.build_pnl(sales_df, commodity_index_df)
        base_revenue = float(base_pnl["net_revenue"].sum())
        base_cogs = float(base_pnl["total_cogs"].sum())
        base_margin = float(base_pnl["gross_margin"].sum())
        base_oi = float(base_pnl["operating_income"].sum())
        other_costs = base_margin - base_oi
        material_fraction = self.settings["financial"]["material_cogs_fraction"]

        def _sim_oi(shock_demand: bool, shock_commodity: bool, shock_fx: bool) -> np.ndarray:
            n = n_simulations
            d = rng.normal(0, demand_vol, n) if shock_demand else np.zeros(n)
            c = rng.standard_t(df=5, size=n) * commodity_vol if shock_commodity else np.zeros(n)
            f = rng.normal(0, fx_vol, n) if shock_fx else np.zeros(n)

            rev = base_revenue * (1 + d) * (1 + f)
            cogs = base_cogs * (1 + d) + base_cogs * material_fraction * c
            oi = rev - cogs - other_costs
            return oi

        # Total
        total_oi = _sim_oi(True, True, True)
        total_var = float(np.var(total_oi))

        # Each component in isolation
        comm_oi = _sim_oi(False, True, False)
        demand_oi = _sim_oi(True, False, False)
        fx_oi = _sim_oi(False, False, True)

        comm_var = float(np.var(comm_oi))
        demand_var = float(np.var(demand_oi))
        fx_var = float(np.var(fx_oi))

        total_measured = comm_var + demand_var + fx_var
        if total_measured < 1e-9:
            return {"commodity_pct": 33.3, "demand_pct": 33.3, "fx_pct": 33.4}

        result = {
            "commodity_pct": round(comm_var / total_measured * 100, 1),
            "demand_pct": round(demand_var / total_measured * 100, 1),
            "fx_pct": round(fx_var / total_measured * 100, 1),
            "total_var": round(total_var, 0),
            "var_95": round(float(np.percentile(total_oi, 5)) - float(np.mean(total_oi)), 0),
            "cvar_95": round(
                float(np.mean(total_oi[total_oi <= np.percentile(total_oi, 5)])) - float(np.mean(total_oi)),
                0,
            ),
            "annual_mean_oi": round(float(np.mean(total_oi)) * len(base_pnl), 0),
        }

        logger.info(
            f"Variance decomposition: commodity={result['commodity_pct']}%, "
            f"demand={result['demand_pct']}%, fx={result['fx_pct']}%"
        )
        return result
