"""
Predictive-Intelligence API routes (Layer 2 surface — G7 + G11).

  GET  /intelligence/change-points/{commodity}   — CUSUM+BOCPD structural-break alert
  GET  /intelligence/change-points               — scan all commodities, flag any needing reforecast
  GET  /intelligence/quantile-var                — gradient-boosted quantile VaR/CVaR
"""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

intelligence_router = APIRouter(prefix="/intelligence", tags=["intelligence"])


def _load_commodity_df():
    from layers.layer1_data.controller import DataLayerController
    return DataLayerController().load_commodity_data()


def _load_commodity_index_df():
    from layers.layer1_data.controller import DataLayerController
    from layers.layer2_intelligence.controller import IntelligenceLayerController
    data = DataLayerController()
    commodity_df = data.load_commodity_data()
    intel = IntelligenceLayerController()
    return intel.generate_commodity_index(commodity_df)


@intelligence_router.get("/change-points/{commodity}")
async def get_change_points(commodity: str, recent_window: int = 6):
    """
    CUSUM + BOCPD structural-break detection for a single commodity.
    Returns alert fields (shifted, direction, confidence) plus full break list.
    """
    try:
        from layers.layer2_intelligence.controller import IntelligenceLayerController
        commodity_df = _load_commodity_df()
        intel = IntelligenceLayerController()
        return intel.detect_structural_breaks(commodity, commodity_df, recent_window=recent_window)
    except Exception as exc:
        logger.warning(f"/intelligence/change-points/{commodity} error: {exc}")
        return {"commodity": commodity, "shifted": False, "error": str(exc)}


@intelligence_router.get("/change-points")
async def scan_all_change_points():
    """
    Scan all 12 commodities for structural breaks.
    Returns only those flagged for reforecast (confidence > 0.6 + shifted).
    Auto-retraining is NOT triggered here (read-only scan).
    """
    try:
        from layers.layer2_intelligence.controller import IntelligenceLayerController
        commodity_df = _load_commodity_df()
        intel = IntelligenceLayerController()
        commodity_cols = [c for c in commodity_df.columns if c != "date"]
        results = {}
        for c in commodity_cols:
            r = intel.detect_structural_breaks(c, commodity_df)
            if r.get("reforecast_recommended") or r.get("n_breaks", 0) > 0:
                results[c] = r
        return {"flagged": results, "total_scanned": len(commodity_cols)}
    except Exception as exc:
        logger.warning(f"/intelligence/change-points error: {exc}")
        return {"flagged": {}, "total_scanned": 0, "error": str(exc)}


@intelligence_router.get("/quantile-var")
async def get_quantile_var(horizon: int = 12):
    """
    Gradient-boosted quantile regression VaR/CVaR on the commodity index.
    Returns asymmetric 5th/50th/95th quantile bands — more accurate than
    symmetric Gaussian VaR for fat-tailed commodity distributions.
    """
    try:
        from layers.layer4_simulation.controller import SimulationLayerController
        commodity_index_df = _load_commodity_index_df()
        sim = SimulationLayerController()
        return sim.quantile_var_forecast(commodity_index_df, horizon=horizon)
    except Exception as exc:
        logger.warning(f"/intelligence/quantile-var error: {exc}")
        return {"error": str(exc), "var_5pct": None, "var_95pct": None}
