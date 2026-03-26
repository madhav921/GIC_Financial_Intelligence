"""
Revenue Drivers (Layer 2 — Driver Engine)

Computes revenue components from demand forecasts and pricing data:
  Revenue = Volume × Net Price
  Net Price = List Price × (1 - Incentive%)

Driver dimensions: Model/Trim, Region, Channel
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings


@dataclass
class RevenueDriverOutput:
    segment: str
    period: str
    gross_revenue: float
    incentive_cost: float
    net_revenue: float
    volume: int
    avg_net_price: float


class RevenueDrivers:
    """
    Calculates revenue drivers from volume and pricing inputs.

    Integrates:
      - Volume from Demand Forecast Model (Layer 3)
      - Price & incentive data from sales systems
      - Elasticity factors from Price Elasticity Model
    """

    def __init__(self):
        self.settings = get_settings()

    def compute(
        self,
        sales_df: pd.DataFrame,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> pd.DataFrame:
        """Compute monthly revenue drivers by segment."""
        df = sales_df.copy()
        if period_start:
            df = df[df["date"] >= period_start]
        if period_end:
            df = df[df["date"] <= period_end]

        # Net price per unit
        df["net_price"] = df["avg_price_usd"] * (1 - df["incentive_pct"])
        df["gross_revenue"] = df["volume"] * df["avg_price_usd"]
        df["incentive_cost"] = df["volume"] * df["avg_price_usd"] * df["incentive_pct"]
        df["net_revenue"] = df["volume"] * df["net_price"]

        # Aggregate by segment and month
        monthly = df.groupby(["date", "segment"]).agg(
            volume=("volume", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            incentive_cost=("incentive_cost", "sum"),
            net_revenue=("net_revenue", "sum"),
            avg_net_price=("net_price", "mean"),
        ).reset_index()

        logger.info(f"Revenue drivers computed: {len(monthly)} segment-months")
        return monthly

    def apply_demand_scenario(
        self,
        revenue_df: pd.DataFrame,
        demand_shock_pct: float = 0.0,
    ) -> pd.DataFrame:
        """Apply a demand shock scenario to revenue projections."""
        df = revenue_df.copy()
        multiplier = 1 + demand_shock_pct
        df["volume"] = (df["volume"] * multiplier).astype(int)
        df["gross_revenue"] *= multiplier
        df["incentive_cost"] *= multiplier
        df["net_revenue"] *= multiplier
        return df

    def summary(self, revenue_df: pd.DataFrame) -> pd.DataFrame:
        """Annual summary by segment."""
        df = revenue_df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        return df.groupby(["year", "segment"]).agg(
            total_volume=("volume", "sum"),
            total_net_revenue=("net_revenue", "sum"),
            avg_net_price=("avg_net_price", "mean"),
        ).reset_index()
