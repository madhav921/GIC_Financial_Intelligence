"""
Variance Bridge Analyzer (Actionable-Intelligence Layer).

Decomposes the EBIT gap between a PLAN scenario and an ACTUAL/current scenario
into an ordered driver waterfall:

    Volume → Price/Mix → Commodity → FX → Warranty → Other (residual)

Attribution logic (standard FP&A bridge conventions):
  • Volume      = Δvolume × plan contribution-margin per unit
  • Price/Mix   = Δ(net revenue) attributable to price & mix at constant volume
  • Commodity   = Δ commodity component of COGS
  • FX          = Δ EBIT attributable to FX translation/transaction
  • Warranty    = -(Δ warranty cost)   (a cost increase is adverse)
  • Other       = residual so the bridge ties out exactly to total variance

If full plan/actual P&L frames are not available, the analyzer accepts summary
dicts and falls back to proportional attribution anchored on the EBIT delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.insights.schemas import VarianceBridge

_DRIVER_ORDER = ["Volume", "Price/Mix", "Commodity", "FX", "Warranty", "Other"]


class VarianceBridgeAnalyzer:
    """Builds EBIT plan→actual variance bridges."""

    @staticmethod
    def _ebit(summary: dict) -> float:
        """Pull an EBIT-like figure from a summary dict, tolerant of key names."""
        for k in ("ebit", "operating_income", "operating_profit", "actual_ebit", "plan_ebit"):
            if k in summary and summary[k] is not None:
                return float(summary[k])
        # Derive from revenue × margin if provided.
        rev = summary.get("net_revenue") or summary.get("revenue")
        mpct = summary.get("operating_margin_pct")
        if rev is not None and mpct is not None:
            return float(rev) * float(mpct) / 100.0
        return 0.0

    def build_bridge(
        self,
        plan_pnl: pd.DataFrame | dict,
        actual_pnl: pd.DataFrame | dict,
        plan_margin_per_unit: float | None = None,
        warranty_delta_gbp: float | None = None,
        commodity_delta_gbp: float | None = None,
        fx_delta_gbp: float | None = None,
    ) -> VarianceBridge:
        """
        Build the variance bridge. ``plan_pnl`` / ``actual_pnl`` may each be a
        monthly P&L DataFrame (as produced by ``FinancialModel.build_pnl``) or a
        summary dict (``{ebit, net_revenue, volume, ...}``).
        """
        plan_s = self._summarise(plan_pnl)
        actual_s = self._summarise(actual_pnl)

        plan_ebit = plan_s["ebit"]
        actual_ebit = actual_s["ebit"]
        total_var = actual_ebit - plan_ebit

        # ── Volume effect ─────────────────────────────────────────────────────
        d_vol = actual_s["volume"] - plan_s["volume"]
        if plan_margin_per_unit is None:
            plan_margin_per_unit = (
                (plan_s["gross_margin"] / plan_s["volume"]) if plan_s["volume"] > 0 else 0.0
            )
        volume_eff = d_vol * plan_margin_per_unit

        # ── Commodity effect ──────────────────────────────────────────────────
        if commodity_delta_gbp is None:
            commodity_delta_gbp = -(actual_s["commodity_cogs"] - plan_s["commodity_cogs"])
        commodity_eff = float(commodity_delta_gbp)

        # ── Warranty effect (cost increase is adverse) ────────────────────────
        if warranty_delta_gbp is None:
            warranty_delta_gbp = actual_s["warranty"] - plan_s["warranty"]
        warranty_eff = -float(warranty_delta_gbp)

        # ── FX effect ─────────────────────────────────────────────────────────
        fx_eff = float(fx_delta_gbp) if fx_delta_gbp is not None else actual_s["fx"] - plan_s["fx"]

        # ── Price/Mix effect ──────────────────────────────────────────────────
        # Revenue delta not explained by volume = price & mix.
        d_rev = actual_s["net_revenue"] - plan_s["net_revenue"]
        price_mix_eff = d_rev - (d_vol * plan_s["avg_price"]) if plan_s["avg_price"] else d_rev

        # ── Other = residual so bridge ties out exactly ───────────────────────
        explained = volume_eff + price_mix_eff + commodity_eff + fx_eff + warranty_eff
        other_eff = total_var - explained

        deltas = {
            "Volume": volume_eff,
            "Price/Mix": price_mix_eff,
            "Commodity": commodity_eff,
            "FX": fx_eff,
            "Warranty": warranty_eff,
            "Other": other_eff,
        }
        return self._assemble(plan_ebit, actual_ebit, total_var, deltas)

    def build_from_scenarios(
        self,
        base_pnl_summary: dict,
        scenario_pnl_summary: dict,
        attribution: dict[str, float] | None = None,
    ) -> VarianceBridge:
        """
        Convenience builder when only EBIT summaries are available.

        ``base_pnl_summary`` is treated as PLAN, ``scenario_pnl_summary`` as ACTUAL.
        Driver shares default to a realistic premium-OEM attribution profile and
        are scaled to tie out exactly to the EBIT delta; pass ``attribution`` to
        override the per-driver shares (must cover the five named drivers).
        """
        plan_ebit = self._ebit(base_pnl_summary)
        actual_ebit = self._ebit(scenario_pnl_summary)
        total_var = actual_ebit - plan_ebit

        # Default attribution shares of total variance (signed share of the gap).
        shares = attribution or {
            "Volume": 0.34,
            "Price/Mix": 0.18,
            "Commodity": 0.30,
            "FX": 0.10,
            "Warranty": 0.08,
        }
        explained = {k: total_var * v for k, v in shares.items()}
        other = total_var - sum(explained.values())
        deltas = {**explained, "Other": other}
        # Ensure canonical ordering & presence.
        deltas = {d: deltas.get(d, 0.0) for d in _DRIVER_ORDER}
        return self._assemble(plan_ebit, actual_ebit, total_var, deltas)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _summarise(self, pnl: pd.DataFrame | dict) -> dict:
        """Reduce a P&L frame or dict to the fields the bridge needs."""
        if isinstance(pnl, dict):
            vol = float(pnl.get("volume", 0.0) or 0.0)
            net_rev = float(pnl.get("net_revenue", pnl.get("revenue", 0.0)) or 0.0)
            return {
                "ebit": self._ebit(pnl),
                "volume": vol,
                "net_revenue": net_rev,
                "gross_margin": float(pnl.get("gross_margin", net_rev * 0.22)),
                "commodity_cogs": float(pnl.get("commodity_cogs", net_rev * 0.349)),
                "warranty": float(pnl.get("warranty_reserve", pnl.get("warranty", net_rev * 0.018))),
                "fx": float(pnl.get("fx", 0.0) or 0.0),
                "avg_price": float(pnl.get("avg_price", (net_rev / vol) if vol > 0 else 0.0)),
            }

        df = pnl
        net_rev = float(df["net_revenue"].sum()) if "net_revenue" in df else 0.0
        vol = float(df["volume"].sum()) if "volume" in df else 0.0
        gm = float(df["gross_margin"].sum()) if "gross_margin" in df else net_rev * 0.22
        cogs = float(df["total_cogs"].sum()) if "total_cogs" in df else net_rev * 0.775
        warranty = (
            float(df["warranty_reserve"].sum()) if "warranty_reserve" in df else net_rev * 0.018
        )
        ebit = (
            float(df["operating_income"].sum())
            if "operating_income" in df
            else gm - warranty
        )
        return {
            "ebit": ebit,
            "volume": vol,
            "net_revenue": net_rev,
            "gross_margin": gm,
            # Material/commodity component of COGS (settings: ~45% of COGS).
            "commodity_cogs": cogs * 0.45,
            "warranty": warranty,
            "fx": 0.0,
            "avg_price": (net_rev / vol) if vol > 0 else 0.0,
        }

    def _assemble(
        self, plan_ebit: float, actual_ebit: float, total_var: float, deltas: dict[str, float]
    ) -> VarianceBridge:
        denom = abs(total_var) if abs(total_var) > 1e-9 else 1.0
        bridge = []
        for driver in _DRIVER_ORDER:
            delta = float(deltas.get(driver, 0.0))
            bridge.append({
                "driver": driver,
                "delta_gbp": round(delta, 2),
                "pct": round(delta / denom * 100, 1),
                "direction": "favourable" if delta >= 0 else "adverse",
                "explanation": self._explain(driver, delta),
            })

        total_pct = (total_var / abs(plan_ebit) * 100) if abs(plan_ebit) > 1e-9 else 0.0
        logger.info(
            f"Variance bridge: plan EBIT={plan_ebit:,.0f} → actual={actual_ebit:,.0f} "
            f"(Δ {total_var:+,.0f}, {total_pct:+.1f}%)"
        )
        return VarianceBridge(
            plan_ebit=round(plan_ebit, 2),
            actual_ebit=round(actual_ebit, 2),
            total_variance_gbp=round(total_var, 2),
            total_variance_pct=round(total_pct, 2),
            bridge=bridge,
        )

    @staticmethod
    def _explain(driver: str, delta: float) -> str:
        good = delta >= 0
        mag = f"£{abs(delta) / 1e6:,.1f}m"
        templates = {
            "Volume": (
                f"Higher delivered volume added {mag} of contribution margin."
                if good else
                f"Soft demand reduced delivered volume, costing {mag} of margin."
            ),
            "Price/Mix": (
                f"Favourable pricing and richer model mix contributed {mag}."
                if good else
                f"Weaker net pricing / adverse mix eroded {mag}."
            ),
            "Commodity": (
                f"Lower raw-material (commodity) costs saved {mag} in COGS."
                if good else
                f"Commodity cost inflation increased COGS by {mag}."
            ),
            "FX": (
                f"FX tailwind added {mag} to EBIT."
                if good else
                f"Adverse FX translation/transaction cost {mag}."
            ),
            "Warranty": (
                f"Lower warranty incidence released {mag} of cost."
                if good else
                f"Elevated warranty claims/accruals added {mag} of cost."
            ),
            "Other": (
                f"Other operating items contributed {mag} (residual)."
                if good else
                f"Other operating items cost {mag} (residual)."
            ),
        }
        return templates.get(driver, f"{driver}: {mag}")
