"""
Cost Drivers (Layer 2 — Driver Engine)

Computes COGS and cost components:
  COGS = f(BOM, Commodity Index)

Driver dimensions: Commodity curves, Capacity utilization, BOM changes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings


class CostDrivers:
    """
    Calculates cost drivers from commodity prices and BOM data.

    Integrates:
      - Commodity Index from Commodity Forecast Model (Layer 3)
      - BOM weights from configuration / SAP
      - Capacity utilization from production data
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_cogs_pct = self.settings["financial"]["base_cogs_pct"]

    def compute_cogs(
        self,
        revenue_df: pd.DataFrame,
        commodity_index_df: pd.DataFrame,
        bom_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute COGS by segment incorporating commodity price movements.

        COGS = Base COGS + Commodity Impact
        Base COGS = Revenue × base_cogs_pct
        Commodity Impact = proportional to commodity index change from baseline
        """
        df = revenue_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Merge commodity index
        ci = commodity_index_df.copy()
        ci["date"] = pd.to_datetime(ci["date"])
        base_index = ci["commodity_index"].iloc[0]
        ci["commodity_impact_pct"] = (ci["commodity_index"] / base_index - 1)

        df = df.merge(ci[["date", "commodity_index", "commodity_impact_pct"]], on="date", how="left")
        df["commodity_impact_pct"] = df["commodity_impact_pct"].fillna(0)

        # Base COGS
        df["base_cogs"] = df["net_revenue"] * self.base_cogs_pct

        # Commodity-driven COGS adjustment
        # If commodity index rises 10%, material costs rise ~10% of the material portion
        material_fraction = self.settings["financial"]["material_cogs_fraction"]
        df["commodity_adjustment"] = df["base_cogs"] * material_fraction * df["commodity_impact_pct"]
        df["total_cogs"] = df["base_cogs"] + df["commodity_adjustment"]

        logger.info(f"COGS computed for {len(df)} rows, avg commodity impact: {df['commodity_impact_pct'].mean():.2%}")
        return df

    def apply_commodity_scenario(
        self,
        cogs_df: pd.DataFrame,
        commodity_shock_pct: float = 0.0,
    ) -> pd.DataFrame:
        """Apply commodity price shock to COGS."""
        df = cogs_df.copy()
        material_fraction = self.settings["financial"]["material_cogs_fraction"]
        additional_impact = df["base_cogs"] * material_fraction * commodity_shock_pct
        df["commodity_adjustment"] += additional_impact
        df["total_cogs"] = df["base_cogs"] + df["commodity_adjustment"]
        return df

    def compute_capacity_cost(
        self,
        production_df: pd.DataFrame,
        fixed_cost_per_unit: float = 5000,
    ) -> pd.DataFrame:
        """
        Compute capacity utilization cost impact.

        Under-utilization increases per-unit fixed cost absorption.
        """
        df = production_df.copy()
        df["utilization_factor"] = df["capacity_utilization_pct"] / 100
        # Fixed cost per unit increases as utilization drops
        df["effective_fixed_cost"] = fixed_cost_per_unit / df["utilization_factor"].clip(lower=0.5)
        df["fixed_cost_total"] = df["effective_fixed_cost"] * df["production_units"]
        return df
