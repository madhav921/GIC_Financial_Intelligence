"""Tests for the financial model and driver engine."""

import pandas as pd
import pytest

from src.drivers.cost_drivers import CostDrivers
from src.drivers.financial_model import FinancialModel
from src.drivers.revenue_drivers import RevenueDrivers
from src.models.commodity_forecast import CommodityForecastModel


class TestRevenueDrivers:
    def test_compute(self, sales_df):
        rd = RevenueDrivers()
        result = rd.compute(sales_df)
        assert "net_revenue" in result.columns
        assert "gross_revenue" in result.columns
        assert result["net_revenue"].sum() > 0
        # Net should be <= gross
        assert result["net_revenue"].sum() <= result["gross_revenue"].sum()

    def test_demand_scenario(self, sales_df):
        rd = RevenueDrivers()
        base = rd.compute(sales_df)
        shocked = rd.apply_demand_scenario(base, demand_shock_pct=-0.10)
        # Revenue should decrease by ~10%
        ratio = shocked["net_revenue"].sum() / base["net_revenue"].sum()
        assert 0.85 < ratio < 0.95


class TestCostDrivers:
    def test_cogs_computation(self, sales_df, commodity_df):
        rd = RevenueDrivers()
        revenue_df = rd.compute(sales_df)

        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        cd = CostDrivers()
        cogs_df = cd.compute_cogs(revenue_df, commodity_index)
        assert "total_cogs" in cogs_df.columns
        assert cogs_df["total_cogs"].sum() > 0

    def test_commodity_scenario(self, sales_df, commodity_df):
        rd = RevenueDrivers()
        revenue_df = rd.compute(sales_df)

        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        cd = CostDrivers()
        base_cogs = cd.compute_cogs(revenue_df, commodity_index)
        shocked_cogs = cd.apply_commodity_scenario(base_cogs, commodity_shock_pct=0.20)
        # COGS should increase with commodity shock
        assert shocked_cogs["total_cogs"].sum() > base_cogs["total_cogs"].sum()


class TestFinancialModel:
    def test_build_pnl(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        fm = FinancialModel()
        pnl = fm.build_pnl(sales_df, commodity_index)

        required_cols = ["net_revenue", "total_cogs", "gross_margin", "operating_income", "net_income"]
        for col in required_cols:
            assert col in pnl.columns

        # Margin = Revenue - COGS
        assert abs(pnl["gross_margin"].sum() - (pnl["net_revenue"].sum() - pnl["total_cogs"].sum())) < 1

    def test_annual_summary(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        fm = FinancialModel()
        pnl = fm.build_pnl(sales_df, commodity_index)
        annual = fm.annual_summary(pnl)

        assert "year" in annual.columns
        assert "gross_margin_pct" in annual.columns

    def test_scenario_pnl(self, sales_df, commodity_df):
        cfm = CommodityForecastModel()
        commodity_index = cfm.generate_commodity_index(commodity_df)

        fm = FinancialModel()
        base = fm.build_pnl(sales_df, commodity_index)
        bear = fm.scenario_pnl(sales_df, commodity_index, demand_shock=-0.10, commodity_shock=0.15)

        # Bear case should have lower margin
        assert bear["gross_margin"].sum() < base["gross_margin"].sum()
