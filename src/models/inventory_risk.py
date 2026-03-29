"""
Inventory & Warranty Risk Model (Layer 3 — Predictive Intelligence)

Estimates:
  - Inventory risk: probability of overstock/stockout by segment
  - Warranty cost risk: expected warranty claims based on production quality signals

Feeds Risk Factors into the Scenario Simulation engine (Layer 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings


@dataclass
class InventoryRiskResult:
    segment: str
    avg_days_of_supply: float
    stockout_probability: float
    overstock_probability: float
    optimal_inventory_units: int
    warranty_rate_pct: float
    expected_warranty_cost_usd: float


class InventoryRiskModel:
    """
    Assesses inventory and warranty risk based on production/sales patterns.
    Uses statistical analysis of inventory turns and warranty claim rates.
    """

    def __init__(self, target_dos: int = 45):
        """
        Args:
            target_dos: Target days of supply (industry standard ~45 for automotive)
        """
        self.target_dos = target_dos
        self.results: dict[str, InventoryRiskResult] = {}

    def analyze(
        self,
        production_df: pd.DataFrame,
        segment: str,
        avg_unit_cost: float = 50000,
        warranty_cost_per_claim: float = 2500,
    ) -> InventoryRiskResult:
        """Analyze inventory and warranty risk for a segment."""
        seg_data = production_df[production_df["segment"] == segment].copy()
        seg_data = seg_data.sort_values("date")

        # Days of supply calculation
        avg_monthly_sales = seg_data["sales_units"].mean()
        daily_sales = avg_monthly_sales / 30
        current_inventory = seg_data["ending_inventory"].iloc[-1]
        avg_inventory = seg_data["ending_inventory"].mean()
        dos = avg_inventory / max(daily_sales, 1)

        # Stockout risk: proportion of months where inventory < 2 weeks of sales
        two_week_threshold = daily_sales * 14
        stockout_months = (seg_data["ending_inventory"] < two_week_threshold).sum()
        stockout_prob = stockout_months / len(seg_data)

        # Overstock risk: proportion of months where inventory > 3 months of sales
        overstock_threshold = avg_monthly_sales * 3
        overstock_months = (seg_data["ending_inventory"] > overstock_threshold).sum()
        overstock_prob = overstock_months / len(seg_data)

        # Optimal inventory (safety stock calculation)
        sales_std = seg_data["sales_units"].std()
        safety_stock = int(1.65 * sales_std)  # 95% service level
        optimal = int(avg_monthly_sales * (self.target_dos / 30) + safety_stock)

        # Warranty analysis
        warranty_rate = seg_data["warranty_claims"].sum() / max(seg_data["sales_units"].sum(), 1) * 100
        monthly_claims = seg_data["warranty_claims"].mean()
        expected_warranty_cost = monthly_claims * 12 * warranty_cost_per_claim

        result = InventoryRiskResult(
            segment=segment,
            avg_days_of_supply=round(dos, 1),
            stockout_probability=round(stockout_prob, 4),
            overstock_probability=round(overstock_prob, 4),
            optimal_inventory_units=optimal,
            warranty_rate_pct=round(warranty_rate, 2),
            expected_warranty_cost_usd=round(expected_warranty_cost, 2),
        )
        self.results[segment] = result

        logger.info(
            f"Inventory risk for {segment}: DOS={dos:.0f}, "
            f"stockout_prob={stockout_prob:.2%}, warranty_rate={warranty_rate:.2f}%"
        )
        return result

    def analyze_all_segments(
        self, production_df: pd.DataFrame
    ) -> dict[str, InventoryRiskResult]:
        settings = get_settings()
        results = {}
        for seg in settings["vehicle_segments"]:
            segment = seg["segment"]
            results[segment] = self.analyze(
                production_df, segment, avg_unit_cost=seg["avg_price_usd"] * 0.68
            )
        return results

    def summary_table(self) -> pd.DataFrame:
        records = []
        for seg, r in self.results.items():
            records.append({
                "segment": seg,
                "avg_days_of_supply": r.avg_days_of_supply,
                "stockout_probability": r.stockout_probability,
                "overstock_probability": r.overstock_probability,
                "optimal_inventory": r.optimal_inventory_units,
                "warranty_rate_pct": r.warranty_rate_pct,
                "annual_warranty_cost": r.expected_warranty_cost_usd,
            })
        return pd.DataFrame(records)
