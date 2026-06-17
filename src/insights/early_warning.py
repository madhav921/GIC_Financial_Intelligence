"""
Early-Warning System (Actionable-Intelligence Layer).

Blends signals from across the platform into a single composite 0–100 risk score
so leadership gets one number ("how worried should we be this month?") plus the
decomposition and the top drivers behind it.

Component weights (sum to 1.0):
  • commodity   (0.30): commodity forecast volatility / model MAPE
  • margin      (0.30): EBIT/gross-margin compression vs plan
  • warranty    (0.20): warranty risk score (from WarrantyModel)
  • demand      (0.20): demand softness vs plan

Each component is normalised to 0–100 (higher = worse) before weighting.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

_WEIGHTS = {"commodity": 0.30, "margin": 0.30, "warranty": 0.20, "demand": 0.20}


class EarlyWarningSystem:
    """Composite cross-domain risk scoring."""

    def compute_risk_score(
        self,
        commodity_volatility_pct: float | None = None,
        commodity_mape_pct: float | None = None,
        margin_compression_pct: float | None = None,
        warranty_risk_score: float | None = None,
        demand_softness_pct: float | None = None,
    ) -> dict:
        """
        Compute the composite risk score. All arguments are optional; missing
        signals fall back to neutral/realistic defaults so the system always
        returns a usable score.

        Args:
            commodity_volatility_pct: annualised commodity index volatility (%).
            commodity_mape_pct:       forecast model MAPE (%) — proxy for uncertainty.
            margin_compression_pct:   EBIT margin compression vs plan (pp, positive=worse).
            warranty_risk_score:      0–100 score from WarrantyModel.warranty_risk_score.
            demand_softness_pct:      demand shortfall vs plan (%, positive=worse).
        """
        # ── Commodity component ───────────────────────────────────────────────
        vol = commodity_volatility_pct if commodity_volatility_pct is not None else 18.0
        mape = commodity_mape_pct if commodity_mape_pct is not None else 9.0
        commodity_comp = float(np.clip(0.6 * (vol / 30.0) + 0.4 * (mape / 15.0), 0, 1) * 100)

        # ── Margin component ──────────────────────────────────────────────────
        compression = margin_compression_pct if margin_compression_pct is not None else 1.2
        margin_comp = float(np.clip(compression / 3.0, 0, 1) * 100)  # 3pp compression → 100

        # ── Warranty component ────────────────────────────────────────────────
        warranty_comp = float(np.clip(
            warranty_risk_score if warranty_risk_score is not None else 45.0, 0, 100
        ))

        # ── Demand component ──────────────────────────────────────────────────
        softness = demand_softness_pct if demand_softness_pct is not None else 4.0
        demand_comp = float(np.clip(softness / 12.0, 0, 1) * 100)  # 12% shortfall → 100

        components = {
            "commodity": round(commodity_comp, 1),
            "margin": round(margin_comp, 1),
            "warranty": round(warranty_comp, 1),
            "demand": round(demand_comp, 1),
        }

        score = sum(_WEIGHTS[k] * v for k, v in components.items())
        score = float(np.clip(score, 0, 100))

        if score >= 75:
            band = "critical"
        elif score >= 55:
            band = "high"
        elif score >= 35:
            band = "elevated"
        else:
            band = "low"

        # Weighted contributions → top drivers.
        contributions = {k: _WEIGHTS[k] * v for k, v in components.items()}
        top_drivers = [
            {"component": k, "contribution": round(v, 1), "score": components[k]}
            for k, v in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        ]

        logger.info(f"Early-warning score={score:.1f} band={band}")
        return {
            "score": round(score, 1),
            "band": band,
            "components": components,
            "weights": _WEIGHTS,
            "top_drivers": top_drivers,
        }
