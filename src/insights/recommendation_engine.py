"""
Recommendation Engine (Actionable-Intelligence Layer — prescriptive core).

Turns quantitative signals (commodity forecasts, exposures, demand elasticity)
into prescriptive, £-quantified actions. The heuristics are deliberately simple,
transparent and documented so a planner can audit every number.

All monetary figures are GBP.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class RecommendationEngine:
    """Heuristic, quantified prescriptive recommendations."""

    # ── 1. Hedging ────────────────────────────────────────────────────────────
    def hedge_recommendation(
        self,
        commodity: str,
        forecast_pct: float,
        exposure_gbp: float,
        current_hedge_ratio: float,
    ) -> dict:
        """
        Recommend a hedge-ratio adjustment for a commodity exposure.

        Formula
        -------
        target_hedge_ratio = clip(base + sensitivity × forecast_move, 0, 0.90)
          • base = 0.50 (a balanced starting hedge book)
          • sensitivity scales with the *upward* price risk: a forecast +X% rise
            pushes the target ratio up (lock in cost), a forecast fall pulls it
            down (stay floating to capture the decline).
        expected_savings_gbp = unhedged_uplift_exposure × forecast_pct
          where unhedged_uplift_exposure = (target − current) × exposure_gbp.
          i.e. the cost increase avoided on the incremental volume now hedged.

        ``forecast_pct`` is a fraction (e.g. 0.18 = +18%).
        """
        base = 0.50
        sensitivity = 1.5
        target = float(np.clip(base + sensitivity * forecast_pct, 0.0, 0.90))
        delta_ratio = target - current_hedge_ratio

        # Savings only accrue when increasing the hedge into a rising market.
        incremental_exposure = max(delta_ratio, 0.0) * exposure_gbp
        expected_savings = incremental_exposure * max(forecast_pct, 0.0)

        if delta_ratio > 0.02:
            action = f"Increase {commodity} hedge ratio to {target:.0%}"
        elif delta_ratio < -0.02:
            action = f"Reduce {commodity} hedge ratio to {target:.0%} (let exposure float)"
        else:
            action = f"Maintain {commodity} hedge ratio near {current_hedge_ratio:.0%}"

        rationale = (
            f"Forecast {commodity} move {forecast_pct:+.1%} over the hedging horizon. "
            f"Recommended target hedge ratio {target:.0%} vs current {current_hedge_ratio:.0%}. "
            f"Locking the incremental £{incremental_exposure / 1e6:,.1f}m of exposure "
            f"avoids ~£{expected_savings / 1e6:,.2f}m of cost inflation."
        )
        logger.info(f"Hedge rec [{commodity}]: target={target:.0%}, savings={expected_savings:,.0f}")
        return {
            "commodity": commodity,
            "action": action,
            "target_hedge_ratio": round(target, 4),
            "current_hedge_ratio": round(float(current_hedge_ratio), 4),
            "expected_savings_gbp": round(expected_savings, 2),
            "rationale": rationale,
        }

    # ── 2. Inventory ──────────────────────────────────────────────────────────
    def inventory_recommendation(
        self,
        commodity: str,
        forecast_pct: float,
        days_of_supply: float,
        target_days: float = 45.0,
    ) -> dict:
        """
        Pre-buy / draw-down advice based on the price forecast and current cover.

        Logic
        -----
        • Rising market (forecast_pct > +5%) and cover below target → PRE-BUY to
          lift cover toward ``target_days`` and beat the increase.
        • Falling market (forecast_pct < −5%) and cover above target → DRAW DOWN
          and defer purchases to buy cheaper later.
        • Otherwise → HOLD.

        ``adjust_days`` quantifies the cover change recommended.
        """
        if forecast_pct > 0.05 and days_of_supply < target_days:
            action = "pre_buy"
            adjust_days = round(target_days - days_of_supply, 1)
            advice = (
                f"Pre-buy {commodity} to lift cover from {days_of_supply:.0f} to "
                f"{target_days:.0f} days ahead of a forecast {forecast_pct:+.1%} rise."
            )
        elif forecast_pct < -0.05 and days_of_supply > target_days:
            action = "draw_down"
            adjust_days = round(days_of_supply - target_days, 1)
            advice = (
                f"Draw down {commodity} inventory toward {target_days:.0f} days and defer "
                f"purchases into a forecast {forecast_pct:+.1%} decline."
            )
        else:
            action = "hold"
            adjust_days = 0.0
            advice = (
                f"Hold {commodity} inventory near {days_of_supply:.0f} days; "
                f"forecast move {forecast_pct:+.1%} does not justify repositioning."
            )

        return {
            "commodity": commodity,
            "action": action,
            "current_days_of_supply": round(float(days_of_supply), 1),
            "target_days_of_supply": round(float(target_days), 1),
            "adjust_days": adjust_days,
            "advice": advice,
        }

    # ── 3. Pricing ────────────────────────────────────────────────────────────
    def pricing_recommendation(
        self,
        segment: str,
        commodity_cost_delta_pct: float,
        elasticity: float,
        cogs_share_of_price: float = 0.55,
    ) -> dict:
        """
        Suggest a price action to protect margin against a commodity cost change.

        Formula
        -------
        cost_per_unit_change% ≈ commodity_cost_delta_pct × cogs_share_of_price
        margin-neutral price change ≈ cost_per_unit_change% (pass-through to hold £ margin)
        Recoverable share is bounded by demand elasticity: highly elastic segments
        cannot fully pass through without volume loss, so we apply a pass-through
        cap = clip(1 / (1 + |elasticity|), 0.3, 1.0).

        ``elasticity`` is the (negative) own-price elasticity of demand.
        """
        cost_change = commodity_cost_delta_pct * cogs_share_of_price
        passthrough_cap = float(np.clip(1.0 / (1.0 + abs(elasticity)), 0.3, 1.0))
        suggested_price_change = cost_change * passthrough_cap

        # Estimated volume response to the price move.
        volume_response = suggested_price_change * elasticity  # elasticity is negative

        if abs(suggested_price_change) < 0.005:
            action = "hold_pricing"
        elif suggested_price_change > 0:
            action = "raise_price"
        else:
            action = "cut_price"

        rationale = (
            f"{segment}: commodity cost moved {commodity_cost_delta_pct:+.1%} "
            f"(≈{cost_change:+.1%} of unit price). With elasticity {elasticity:.2f}, "
            f"recover ~{passthrough_cap:.0%} via a {suggested_price_change:+.1%} price action; "
            f"expected volume response ~{volume_response:+.1%}."
        )
        return {
            "segment": segment,
            "action": action,
            "suggested_price_change_pct": round(suggested_price_change * 100, 2),
            "passthrough_pct": round(passthrough_cap * 100, 1),
            "expected_volume_change_pct": round(volume_response * 100, 2),
            "rationale": rationale,
        }
