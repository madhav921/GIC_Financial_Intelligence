"""
Commodity → P&L Shock Calculator.

Real-time computation of the P&L cascade from commodity price shocks.
Powers the live dashboard sliders.

Core formula:
    Commodity_Impact = Base_COGS × Material_Fraction × BOM_Weight[commodity] × Shock_Pct
    After_Tax_EBIT_Impact = -Commodity_Impact × (1 - Tax_Rate)

For a £20B revenue JLR:
    Steel +10% → EBIT impact ~ -£62M
    Lithium +20% → EBIT impact ~ -£96M (EV-heavy BOM)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import get_settings


class CommodityShockCalculator:
    """
    Given a commodity price shock (%), computes the full P&L cascade.
    Used by dashboard live sliders and the API /pnl/shock endpoint.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings or get_settings()
        self.bom_weights = self._load_bom_weights()

    def _load_bom_weights(self) -> dict[str, float]:
        """
        Extract BOM weights from settings.yaml commodities list.
        Keys are lowercased commodity names for case-insensitive lookup.
        Weights are raw (may not sum to 1.0) — normalised per-commodity in compute_shock.
        """
        weights: dict[str, float] = {}
        for c in self._settings.get("commodities", []):
            key = c["name"].lower().replace(" ", "_")
            weights[key] = float(c.get("bom_weight", 0.0))
        return weights

    def _resolve_key(self, commodity: str) -> str:
        """Normalise commodity name to internal lowercase key."""
        return commodity.lower().replace(" ", "_").replace("-", "_")

    def compute_shock(
        self,
        commodity: str,
        shock_pct: float,
        base_revenue: float,
        base_cogs_pct: float | None = None,
        material_fraction: float | None = None,
        tax_rate: float | None = None,
    ) -> dict:
        """
        Compute the P&L impact of a commodity price shock.

        Args:
            commodity:        Commodity name (case-insensitive, e.g. "Lithium")
            shock_pct:        Fractional price change (e.g. 0.20 = +20%)
            base_revenue:     Annual base revenue (£ or $)
            base_cogs_pct:    COGS as % of revenue (default: from settings)
            material_fraction: Materials as % of COGS (default: from settings)
            tax_rate:         Corporate tax rate (default: from settings)

        Returns:
            dict with:
                commodity       : str
                shock_pct       : float
                cogs_impact     : float  (£ change in COGS; positive = cost up)
                ebit_impact     : float  (£ change in after-tax EBIT; negative = bad)
                margin_impact_bps: float (basis point change in gross margin %)
                pct_of_base_ebit: float  (impact as % of ~10% EBIT base)
        """
        fin = self._settings.get("financial", {})
        _base_cogs_pct = base_cogs_pct if base_cogs_pct is not None else fin.get("base_cogs_pct", 0.775)
        _material_fraction = material_fraction if material_fraction is not None else fin.get("material_cogs_fraction", 0.45)
        _tax_rate = tax_rate if tax_rate is not None else fin.get("tax_rate", 0.19)

        key = self._resolve_key(commodity)
        bom_weight = self.bom_weights.get(key, 0.0)

        if bom_weight == 0.0:
            logger.warning(f"CommodityShockCalculator: BOM weight for '{commodity}' not found — impact = 0")

        base_cogs = base_revenue * _base_cogs_pct
        material_cogs = base_cogs * _material_fraction

        # Cost impact from this commodity's BOM share
        cogs_impact = material_cogs * bom_weight * shock_pct

        # EBIT: COGS up → operating income down
        ebit_before_tax = -cogs_impact
        after_tax_impact = ebit_before_tax * (1 - _tax_rate)

        # Basis points change in gross margin %
        margin_impact_bps = (-cogs_impact / base_revenue) * 10_000 if base_revenue > 0 else 0.0

        # As % of estimated base EBIT (assume ~8% operating margin for JLR)
        base_ebit = base_revenue * 0.08
        pct_of_base_ebit = (after_tax_impact / base_ebit * 100) if base_ebit > 0 else 0.0

        return {
            "commodity": commodity,
            "shock_pct": shock_pct,
            "cogs_impact": round(cogs_impact, 0),
            "ebit_impact": round(after_tax_impact, 0),
            "margin_impact_bps": round(margin_impact_bps, 2),
            "pct_of_base_ebit": round(pct_of_base_ebit, 2),
        }

    def waterfall(
        self,
        shocks: dict[str, float],
        base_revenue: float,
    ) -> list[dict]:
        """
        Compute waterfall decomposition for multiple simultaneous commodity shocks.

        Args:
            shocks:       {commodity_name: shock_pct} e.g. {"Lithium": 0.20, "Steel": -0.05}
            base_revenue: Annual base revenue

        Returns:
            List of shock result dicts, sorted by |ebit_impact| descending.
            Only includes commodities with non-zero shocks.
        """
        results = []
        for commodity, shock_pct in shocks.items():
            if abs(shock_pct) < 1e-6:
                continue
            result = self.compute_shock(commodity, shock_pct, base_revenue)
            results.append(result)

        # Sort by absolute EBIT impact (biggest first)
        results.sort(key=lambda x: abs(x["ebit_impact"]), reverse=True)

        total_cogs = sum(r["cogs_impact"] for r in results)
        total_ebit = sum(r["ebit_impact"] for r in results)
        logger.info(
            f"Waterfall: {len(results)} commodities, "
            f"total COGS impact = {total_cogs:+,.0f}, "
            f"EBIT impact = {total_ebit:+,.0f}"
        )
        return results

    def total_ebit_impact(self, shocks: dict[str, float], base_revenue: float) -> float:
        """Convenience: total EBIT impact across all shocks."""
        return sum(
            self.compute_shock(c, s, base_revenue)["ebit_impact"]
            for c, s in shocks.items()
            if abs(s) > 1e-6
        )
