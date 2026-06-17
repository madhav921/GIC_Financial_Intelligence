"""
Layer 3: Financial Drivers Controller

Single entry point for deterministic financial modeling and P&L construction.
Delegates to src/drivers/ modules.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger

from src.drivers.financial_model import FinancialModel


class FinancialLayerController:
    """
    Layer 3 Controller — Financial Drivers

    Usage:
        fin = FinancialLayerController()
        pnl      = fin.build_pnl(sales_df, commodity_index_df)
        scenario = fin.apply_scenario(sales_df, commodity_index_df, demand_shock=-0.08)
        shocked  = fin.apply_commodity_shock(pnl, {"Lithium": 0.15, "Steel": 0.05})
        annual   = fin.annual_summary(pnl)
    """

    def __init__(self):
        self._model = FinancialModel()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_pnl(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        production_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build monthly P&L from driver inputs.
        Returns DataFrame: date, segment, volume, net_revenue, total_cogs,
                           gross_margin, warranty_reserve, depreciation,
                           operating_income, operating_margin_pct, tax, net_income
        """
        logger.info("Layer 3: Building monthly P&L")
        pnl = self._model.build_pnl(sales_df, commodity_index_df, production_df)
        logger.info(
            f"Layer 3: P&L built — {len(pnl)} rows, "
            f"total revenue: {pnl['net_revenue'].sum():,.0f}"
        )
        return pnl

    def apply_scenario(
        self,
        sales_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        demand_shock: float = 0.0,
        commodity_shock: float = 0.0,
    ) -> pd.DataFrame:
        """
        Build P&L under a specific scenario.
        demand_shock: fractional change in demand (e.g., -0.08 = -8%)
        commodity_shock: fractional change in commodity costs (e.g., 0.40 = +40%)
        """
        logger.info(
            f"Layer 3: Scenario P&L — demand={demand_shock:+.1%}, "
            f"commodity={commodity_shock:+.1%}"
        )
        return self._model.scenario_pnl(sales_df, commodity_index_df, demand_shock, commodity_shock)

    def apply_commodity_shock(
        self,
        base_pnl: pd.DataFrame,
        shocks: dict[str, float],
    ) -> pd.DataFrame:
        """
        Inject per-commodity price shocks into a base P&L and recompute EBIT.
        shocks: {commodity_name: fractional_shock}
        e.g. {"Lithium": 0.20, "Steel": -0.05}
        Recalculation time: <1 second.
        """
        logger.info(f"Layer 3: Applying commodity shocks — {shocks}")
        return self._model.apply_commodity_shock(base_pnl, shocks)

    def annual_summary(self, pnl_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly P&L to annual summary by year and segment."""
        return self._model.annual_summary(pnl_df)
