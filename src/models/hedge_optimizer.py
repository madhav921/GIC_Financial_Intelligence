"""
Hedge Optimizer — optimal commodity hedge ratio calculation.

Combines:
  - ML commodity price forecast (mean + uncertainty from confidence interval)
  - Current futures curve price
  - Hedge cost (basis points)

to compute the hedge ratio h* ∈ [0, 1] that minimises a blend of:
  - Expected procurement cost  (want to minimise)
  - Value at Risk at 95%       (want to cap downside)

Core formula:
    cost(h) = (1-h)·E[market_price]·units + h·futures_price·(1 + cost_bps)·units
    VaR(h)  = (1-h)·(E[price] + 1.645·σ)·units + h·futures_price·(1+cost_bps)·units

Optimal h* = argmin [α·cost(h) + (1-α)·VaR(h)],  α=0.5 by default.

Result interpretation:
  hedge_ratio=0.62 → hedge 62% of exposure through futures/forwards
  expected_savings → E[cost_unhedged] - E[cost_hedged] in currency units
  var_reduction    → VaR_unhedged - VaR_hedged
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class HedgeOptimizer:
    """
    Portfolio-theory hedge sizing for a single commodity exposure.

    Can be called per-commodity per-month for a full hedge schedule,
    or with aggregate annual exposure for strategic planning.
    """

    def optimize(
        self,
        forecast_mean: float,
        forecast_std: float,
        futures_price: float,
        exposure_units: float,
        hedge_cost_bps: float = 30.0,
        confidence: float = 0.95,
        alpha: float = 0.5,
    ) -> dict:
        """
        Compute the optimal hedge ratio.

        Args:
            forecast_mean:   ML point forecast of the commodity price
            forecast_std:    Uncertainty (σ) — typically CI_width / (2 × z_score)
            futures_price:   Current futures/forward price for the hedge
            exposure_units:  Physical quantity exposed (tonnes, kg, etc.)
            hedge_cost_bps:  Cost of hedging in basis points (default 30bps = 0.30%)
            confidence:      VaR confidence level (default 0.95)
            alpha:           Weight on expected cost vs VaR (0=pure VaR, 1=pure cost)

        Returns:
            dict with:
                optimal_hedge_ratio : float in [0, 1]
                expected_savings    : float (currency, unhedged minus hedged expected cost)
                var_reduction       : float (currency, VaR reduction from hedging)
                hedge_cost          : float (currency, cost of the hedge itself)
                recommendation      : str   (human-readable)
        """
        # z-score for one-tailed VaR
        z = float(np.abs(np.percentile(np.random.standard_normal(100_000), (1 - confidence) * 100)))

        hedge_cost_frac = hedge_cost_bps / 10_000.0

        def expected_cost(h: float) -> float:
            """Expected total procurement cost at hedge ratio h."""
            unhedged = exposure_units * (1.0 - h) * forecast_mean
            hedged = exposure_units * h * futures_price * (1.0 + hedge_cost_frac)
            return unhedged + hedged

        def value_at_risk(h: float) -> float:
            """VaR at confidence level — worst-case procurement cost."""
            worst_price = forecast_mean + z * forecast_std
            unhedged = exposure_units * (1.0 - h) * worst_price
            hedged = exposure_units * h * futures_price * (1.0 + hedge_cost_frac)
            return unhedged + hedged

        def objective(h: float) -> float:
            return alpha * expected_cost(h) + (1.0 - alpha) * value_at_risk(h)

        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        h_star = float(np.clip(result.x, 0.0, 1.0))

        unhedged_expected = expected_cost(0.0)
        hedged_expected = expected_cost(h_star)
        unhedged_var = value_at_risk(0.0)
        hedged_var = value_at_risk(h_star)

        cost_of_hedge = exposure_units * h_star * futures_price * hedge_cost_frac
        expected_savings = unhedged_expected - hedged_expected
        var_reduction = unhedged_var - hedged_var

        recommendation = (
            f"Hedge {h_star * 100:.0f}% of exposure "
            f"(saves ~{expected_savings:+,.0f} expected, "
            f"reduces VaR by ~{var_reduction:,.0f})"
        )

        return {
            "optimal_hedge_ratio": round(h_star, 3),
            "expected_savings": round(expected_savings, 0),
            "var_reduction": round(var_reduction, 0),
            "hedge_cost": round(cost_of_hedge, 0),
            "unhedged_expected_cost": round(unhedged_expected, 0),
            "hedged_expected_cost": round(hedged_expected, 0),
            "recommendation": recommendation,
        }

    def schedule(
        self,
        monthly_forecasts: list[float],
        monthly_stds: list[float],
        monthly_futures: list[float],
        monthly_exposure: list[float],
        hedge_cost_bps: float = 30.0,
    ) -> list[dict]:
        """
        Compute optimal hedge ratios for each month in a forecast horizon.

        Args:
            monthly_forecasts: List of monthly price forecasts
            monthly_stds:      List of monthly forecast uncertainties
            monthly_futures:   List of monthly futures prices
            monthly_exposure:  List of monthly physical exposure (units)

        Returns:
            List of per-month hedge results.
        """
        n = min(len(monthly_forecasts), len(monthly_stds), len(monthly_futures), len(monthly_exposure))
        results = []
        for i in range(n):
            result = self.optimize(
                forecast_mean=monthly_forecasts[i],
                forecast_std=monthly_stds[i],
                futures_price=monthly_futures[i],
                exposure_units=monthly_exposure[i],
                hedge_cost_bps=hedge_cost_bps,
            )
            result["month"] = i + 1
            results.append(result)
        return results
