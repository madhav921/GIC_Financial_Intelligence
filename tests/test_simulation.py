"""Tests for Monte Carlo simulation and scenario engine."""

import pytest

from src.models.commodity_forecast import CommodityForecastModel
from src.simulation.monte_carlo import MonteCarloEngine
from src.simulation.scenario_engine import ScenarioDefinition, ScenarioEngine


class TestMonteCarloEngine:
    def test_basic_simulation(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        mc = MonteCarloEngine()
        result = mc.run(
            sales_df, commodity_index,
            scenario_name="test",
            n_simulations=500,
        )

        assert result.n_simulations == 500
        assert len(result.net_revenue_dist) == 500
        assert len(result.gross_margin_dist) == 500

        stats = result.compute_stats()
        assert "gross_margin" in stats
        assert stats["gross_margin"]["mean"] > 0
        assert stats["gross_margin"]["p5"] < stats["gross_margin"]["p95"]

    def test_var_calculation(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        mc = MonteCarloEngine()
        result = mc.run(sales_df, commodity_index, n_simulations=1000)
        mar = result.margin_at_risk(0.95)
        assert mar < result.stats["gross_margin"]["mean"]


class TestScenarioEngine:
    def test_run_scenario(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        engine = ScenarioEngine()
        scenario = ScenarioDefinition(
            name="test_bear",
            description="Test bear scenario",
            demand_shock=-0.10,
            commodity_shock=0.15,
        )
        result = engine.run_scenario(
            scenario, sales_df, commodity_index,
            run_monte_carlo=False,
        )

        assert "deterministic_pnl" in result
        assert "annual_summary" in result

    def test_preset_scenarios(self):
        presets = ScenarioEngine.preset_scenarios()
        assert len(presets) >= 5
        names = [s.name for s in presets]
        assert "Base Case" in names
        assert "Commodity Crisis" in names

    def test_compare_scenarios(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        engine = ScenarioEngine()
        scenarios = [
            ScenarioDefinition("base", ""),
            ScenarioDefinition("bear", "", demand_shock=-0.10, commodity_shock=0.20),
        ]
        comparison = engine.compare_scenarios(scenarios, sales_df, commodity_index)

        assert len(comparison) == 2
        # Bear should have lower margin
        base_margin = comparison[comparison["scenario"] == "base"]["total_margin"].iloc[0]
        bear_margin = comparison[comparison["scenario"] == "bear"]["total_margin"].iloc[0]
        assert bear_margin < base_margin
