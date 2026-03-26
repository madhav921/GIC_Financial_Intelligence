"""
Scenario Engine & What-If Interface (Layer 4)

Provides a high-level interface for running what-if scenarios:
  - "What if lithium +15%?"
  - "What if demand -8% EU?"
  - "What if interest rate cuts occur?"

Translates natural-language–style scenario parameters into
quantitative shocks and runs them through the financial model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.drivers.financial_model import FinancialModel
from src.simulation.monte_carlo import MonteCarloEngine, SimulationResult


@dataclass
class ScenarioDefinition:
    """A user-defined what-if scenario."""
    name: str
    description: str
    demand_shock: float = 0.0           # e.g., -0.08 for -8%
    commodity_shock: float = 0.0        # e.g., 0.15 for +15%
    fx_shock: float = 0.0              # e.g., 0.05 for +5%
    specific_commodity: str | None = None  # e.g., "Lithium"
    specific_region: str | None = None     # e.g., "EU"

    def summary(self) -> str:
        parts = [f"Scenario: {self.name}"]
        if self.demand_shock:
            region = f" ({self.specific_region})" if self.specific_region else ""
            parts.append(f"  Demand: {self.demand_shock:+.1%}{region}")
        if self.commodity_shock:
            comm = f" ({self.specific_commodity})" if self.specific_commodity else " (all)"
            parts.append(f"  Commodity: {self.commodity_shock:+.1%}{comm}")
        if self.fx_shock:
            parts.append(f"  FX: {self.fx_shock:+.1%}")
        return "\n".join(parts)


class ScenarioEngine:
    """
    High-level scenario analysis engine.

    Combines deterministic financial model with Monte Carlo simulation
    to provide both point estimates and probability distributions
    for each scenario.
    """

    def __init__(self):
        self.financial_model = FinancialModel()
        self.mc_engine = MonteCarloEngine()
        self.scenario_history: list[dict] = []

    def run_scenario(
        self,
        scenario: ScenarioDefinition,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        run_monte_carlo: bool = True,
        n_simulations: int = 5000,
    ) -> dict:
        """
        Execute a scenario and return deterministic + probabilistic results.

        Returns:
            dict with keys: 'deterministic_pnl', 'simulation', 'scenario'
        """
        logger.info(f"Running scenario: {scenario.name}")
        logger.info(scenario.summary())

        # Deterministic P&L
        det_pnl = self.financial_model.scenario_pnl(
            sales_df=sales_df,
            commodity_index_df=commodity_index_df,
            demand_shock=scenario.demand_shock,
            commodity_shock=scenario.commodity_shock,
        )

        result = {
            "scenario": scenario,
            "deterministic_pnl": det_pnl,
            "annual_summary": self.financial_model.annual_summary(det_pnl),
        }

        # Monte Carlo overlay
        if run_monte_carlo:
            sim_result = self.mc_engine.run(
                sales_df=sales_df,
                commodity_index_df=commodity_index_df,
                scenario_name=scenario.name,
                n_simulations=n_simulations,
                demand_mean=scenario.demand_shock,
                commodity_mean=scenario.commodity_shock,
            )
            result["simulation"] = sim_result

        # Track history
        self.scenario_history.append({
            "scenario_name": scenario.name,
            "demand_shock": scenario.demand_shock,
            "commodity_shock": scenario.commodity_shock,
            "total_revenue": det_pnl["net_revenue"].sum(),
            "total_margin": det_pnl["gross_margin"].sum(),
        })

        return result

    def compare_scenarios(
        self,
        scenarios: list[ScenarioDefinition],
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run and compare multiple scenarios side by side."""
        records = []
        for scenario in scenarios:
            result = self.run_scenario(
                scenario, sales_df, commodity_index_df,
                run_monte_carlo=False,
            )
            pnl = result["deterministic_pnl"]
            records.append({
                "scenario": scenario.name,
                "total_revenue": pnl["net_revenue"].sum(),
                "total_cogs": pnl["total_cogs"].sum(),
                "total_margin": pnl["gross_margin"].sum(),
                "margin_pct": pnl["gross_margin"].sum() / pnl["net_revenue"].sum() * 100
                    if pnl["net_revenue"].sum() != 0 else 0,
                "operating_income": pnl["operating_income"].sum(),
            })
        return pd.DataFrame(records).round(0)

    @staticmethod
    def preset_scenarios() -> list[ScenarioDefinition]:
        """Return a set of standard what-if scenarios."""
        return [
            ScenarioDefinition("Base Case", "No shocks"),
            ScenarioDefinition("Lithium +15%", "Lithium price spike",
                             commodity_shock=0.15, specific_commodity="Lithium"),
            ScenarioDefinition("EU Demand -8%", "European demand contraction",
                             demand_shock=-0.08, specific_region="EU"),
            ScenarioDefinition("Commodity Crisis", "Broad commodity surge",
                             commodity_shock=0.40, demand_shock=-0.05),
            ScenarioDefinition("Bull Market", "Strong demand + low costs",
                             demand_shock=0.10, commodity_shock=-0.05),
            ScenarioDefinition("Rate Cuts", "Interest rate reduction stimulus",
                             demand_shock=0.05, commodity_shock=-0.02),
            ScenarioDefinition("Stagflation", "Low demand + high costs",
                             demand_shock=-0.12, commodity_shock=0.25, fx_shock=0.08),
        ]
