"""
Capital Drivers (Layer 2 — Driver Engine)

Tracks capital expenditure, utilization, and depreciation:
  - Capex timing and phasing
  - Plant utilization rates
  - Depreciation logic (straight-line)

Feeds into financial model for cash flow and balance sheet projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings


class CapitalDrivers:
    """
    Capital expenditure modelling for plant and equipment.

    Placeholder with configurable parameters — real implementation
    would pull from SAP Asset Accounting (AA) module.
    """

    def __init__(self):
        self.settings = get_settings()
        self.depreciation_years = self.settings["financial"]["depreciation_years"]

    def compute_depreciation_schedule(
        self,
        capex_items: list[dict],
    ) -> pd.DataFrame:
        """
        Generate straight-line depreciation schedule for capex items.

        Args:
            capex_items: List of dicts with keys:
                - name: asset name
                - amount: capex amount USD
                - start_date: when asset enters service
                - useful_life_years: override default depreciation period
        """
        records = []
        for item in capex_items:
            amount = item["amount"]
            life = item.get("useful_life_years", self.depreciation_years)
            monthly_dep = amount / (life * 12)
            start = pd.Timestamp(item["start_date"])

            for month in range(life * 12):
                date = start + pd.DateOffset(months=month)
                accumulated = monthly_dep * (month + 1)
                nbv = amount - accumulated
                records.append({
                    "date": date,
                    "asset": item["name"],
                    "monthly_depreciation": round(monthly_dep, 2),
                    "accumulated_depreciation": round(accumulated, 2),
                    "net_book_value": round(max(nbv, 0), 2),
                })

        return pd.DataFrame(records)

    def sample_capex_plan(self) -> list[dict]:
        """Generate a sample capex plan for development/testing."""
        return [
            {"name": "EV_Battery_Plant", "amount": 500_000_000, "start_date": "2024-01-01", "useful_life_years": 10},
            {"name": "Paint_Shop_Upgrade", "amount": 120_000_000, "start_date": "2024-06-01", "useful_life_years": 8},
            {"name": "Robotics_Line_3", "amount": 80_000_000, "start_date": "2025-01-01", "useful_life_years": 7},
            {"name": "Stamping_Press", "amount": 45_000_000, "start_date": "2025-03-01", "useful_life_years": 12},
        ]
