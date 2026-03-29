"""
Deterministic Financial Driver Model

Codified financial logic translating drivers into financial outcomes:
  Revenue  = Volume × Net Price
  COGS     = f(BOM, Commodity Index)
  Margin   = Revenue - COGS
  Inventory = Production - Sales

This is the core financial calculation engine that sits between
the AI models (Layer 3) and the Scenario Simulation (Layer 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings
from src.drivers.capital_drivers import CapitalDrivers
from src.drivers.cost_drivers import CostDrivers
from src.drivers.revenue_drivers import RevenueDrivers


@dataclass
class FinancialSummary:
    """P&L summary for a period."""
    period: str
    net_revenue: float
    total_cogs: float
    gross_margin: float
    gross_margin_pct: float
    warranty_reserve: float
    depreciation: float
    operating_income: float
    operating_margin_pct: float
    tax: float
    net_income: float


class FinancialModel:
    """
    Deterministic financial model that combines all driver outputs
    into a unified P&L, cash flow, and margin analysis.
    """

    def __init__(self):
        self.settings = get_settings()
        self.revenue_drivers = RevenueDrivers()
        self.cost_drivers = CostDrivers()
        self.capital_drivers = CapitalDrivers()

    def _apply_pnl_items(self, pnl: pd.DataFrame) -> pd.DataFrame:
        """Apply warranty, depreciation, tax and net income to a gross-margin DataFrame."""
        pnl["gross_margin"] = pnl["net_revenue"] - pnl["total_cogs"]
        pnl["gross_margin_pct"] = np.where(
            pnl["net_revenue"] > 0,
            pnl["gross_margin"] / pnl["net_revenue"] * 100,
            0,
        )

        warranty_pct = self.settings["financial"]["warranty_reserve_pct"]
        pnl["warranty_reserve"] = pnl["net_revenue"] * warranty_pct

        capex_plan = self.capital_drivers.sample_capex_plan()
        total_annual_dep = sum(item["amount"] / item.get("useful_life_years", 7) for item in capex_plan)
        pnl["depreciation"] = total_annual_dep / 12

        pnl["operating_income"] = pnl["gross_margin"] - pnl["warranty_reserve"] - pnl["depreciation"]
        pnl["operating_margin_pct"] = np.where(
            pnl["net_revenue"] > 0,
            pnl["operating_income"] / pnl["net_revenue"] * 100,
            0,
        )

        tax_rate = self.settings["financial"]["tax_rate"]
        pnl["tax"] = np.where(pnl["operating_income"] > 0, pnl["operating_income"] * tax_rate, 0)
        pnl["net_income"] = pnl["operating_income"] - pnl["tax"]
        return pnl

    def build_pnl(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        production_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build a monthly P&L statement from driver inputs.

        Returns a DataFrame with revenue, COGS, margin, and operating income.
        """
        # Step 1: Revenue
        revenue_df = self.revenue_drivers.compute(sales_df)

        # Step 2: COGS (with commodity impact)
        cogs_df = self.cost_drivers.compute_cogs(revenue_df, commodity_index_df)

        # Step 3: Build P&L
        pnl = cogs_df.copy()
        pnl = self._apply_pnl_items(pnl)

        logger.info(f"P&L built: {len(pnl)} rows, total net revenue: ${pnl['net_revenue'].sum():,.0f}")
        return pnl

    def annual_summary(self, pnl_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly P&L to annual summary."""
        df = pnl_df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year

        annual = df.groupby(["year", "segment"]).agg(
            volume=("volume", "sum"),
            net_revenue=("net_revenue", "sum"),
            total_cogs=("total_cogs", "sum"),
            gross_margin=("gross_margin", "sum"),
            warranty_reserve=("warranty_reserve", "sum"),
            operating_income=("operating_income", "sum"),
            net_income=("net_income", "sum"),
        ).reset_index()

        annual["gross_margin_pct"] = np.where(
            annual["net_revenue"] > 0,
            annual["gross_margin"] / annual["net_revenue"] * 100,
            0,
        )
        annual["operating_margin_pct"] = np.where(
            annual["net_revenue"] > 0,
            annual["operating_income"] / annual["net_revenue"] * 100,
            0,
        )

        return annual.round(2)

    def scenario_pnl(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        demand_shock: float = 0.0,
        commodity_shock: float = 0.0,
    ) -> pd.DataFrame:
        """Build P&L under a specific scenario."""
        # Apply demand shock
        revenue_df = self.revenue_drivers.compute(sales_df)
        if demand_shock != 0:
            revenue_df = self.revenue_drivers.apply_demand_scenario(revenue_df, demand_shock)

        # Apply commodity shock to COGS
        cogs_df = self.cost_drivers.compute_cogs(revenue_df, commodity_index_df)
        if commodity_shock != 0:
            cogs_df = self.cost_drivers.apply_commodity_scenario(cogs_df, commodity_shock)

        # Build P&L from adjusted data
        pnl = cogs_df.copy()
        pnl = self._apply_pnl_items(pnl)
        return pnl
