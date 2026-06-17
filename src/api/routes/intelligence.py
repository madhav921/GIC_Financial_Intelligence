"""
Predictive-Intelligence API routes (Layer 2 surface — G7 + G11).

  GET  /intelligence/change-points/{commodity}   — CUSUM+BOCPD structural-break alert
  GET  /intelligence/change-points               — scan all commodities, flag any needing reforecast
  GET  /intelligence/audit                       — real JSONL audit trail events
  GET  /intelligence/bias                        — per-commodity bias and MAPE metrics (live)
  GET  /intelligence/narrative/{commodity}       — JLR CFO narrative (fast, uses pipeline cache)
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


@intelligence_router.get("/audit")
async def get_audit_trail(limit: int = 50, event_type: str | None = None):
    """
    Real audit events from the governance JSONL trail.
    Returns an empty list when no events have been logged yet.
    """
    try:
        from src.governance.audit_trail import AuditTrail
        trail = AuditTrail()
        entries = trail.get_entries(event_type=event_type, limit=limit)
        return {"events": entries, "total": len(entries), "source": "live"}
    except Exception as exc:
        logger.warning(f"/intelligence/audit error: {exc}")
        return {"events": [], "total": 0, "source": "error", "error": str(exc)}


@intelligence_router.get("/bias")
async def get_bias_metrics():
    """
    Per-commodity forecast bias and MAPE from the pipeline cache.

    Uses a naive 3-month rolling-mean forecast vs actuals on the same
    commodity data used by every other route. Triggers a cache warm-up
    if the cache is cold.
    """
    try:
        from src.api.pipeline_cache import get_snapshot, trigger_refresh
        snap = get_snapshot()
        bias = snap.get("bias_metrics")
        if not bias:
            trigger_refresh()
            return {"metrics": [], "source": "warming", "refreshed_at": None}
        return {
            "metrics": bias,
            "source": "live",
            "refreshed_at": snap.get("refreshed_at"),
        }
    except Exception as exc:
        logger.warning(f"/intelligence/bias error: {exc}")
        return {"metrics": [], "source": "error", "error": str(exc)}


_llm_engine = None  # module-level singleton — initialised once


def _get_llm_engine():
    global _llm_engine
    if _llm_engine is None:
        from layers.layer5_governance.llm_engine import GICLLMEngine
        _llm_engine = GICLLMEngine()
    return _llm_engine


@intelligence_router.get("/narrative/{commodity}")
async def get_commodity_narrative(commodity: str):
    """
    JLM CFO-grade procurement intelligence narrative for a commodity.

    Uses live pipeline cache data (same DataLayerController source as every other route).
    No SARIMAX run needed — typically responds in <100ms on the template backend.
    Normalises space/underscore in commodity name (e.g. 'Natural Gas' → 'Natural_Gas').
    """
    try:
        from src.api.pipeline_cache import get_snapshot, trigger_refresh
        from layers.layer5_governance.llm_engine import _COMMODITY_CONTEXT

        snap = get_snapshot()
        if not snap.get("forecasts"):
            trigger_refresh()

        forecasts = snap.get("forecasts") or {}
        # Try exact match, then underscore variant, then space variant
        commodity_under = commodity.replace(" ", "_")
        commodity_space = commodity.replace("_", " ")
        fc = (
            forecasts.get(commodity)
            or forecasts.get(commodity_under)
            or forecasts.get(commodity_space)
            or {}
        )
        # Canonical name for context lookup: try both variants
        ctx = (
            _COMMODITY_CONTEXT.get(commodity)
            or _COMMODITY_CONTEXT.get(commodity_under)
            or _COMMODITY_CONTEXT.get(commodity_space)
            or {}
        )

        forecast_pct = float(fc.get("forecast_pct", 0.0))
        exposure_gbp = float(fc.get("exposure_gbp", 0.0))
        hedge_ratio = float(fc.get("hedge_ratio", 0.40))
        drivers = ctx.get("drivers", ["market supply-demand", "macro conditions", "energy costs"])

        # MAPE from bias metrics (commodity name may use underscore in bias list)
        mape = None
        for b in (snap.get("bias_metrics") or []):
            bc = b.get("commodity", "")
            if bc == commodity or bc == commodity_under or bc == commodity_space:
                mape = b.get("mape")
                break

        llm = _get_llm_engine()
        # Use the display name (spaces) for narrative readability
        display_name = commodity_space if "_" in commodity else commodity
        narrative = llm.explain_forecast(
            commodity=display_name,
            forecast_pct=forecast_pct,
            drivers=drivers,
            exposure_gbp=exposure_gbp,
            hedge_ratio=hedge_ratio,
            mape=mape,
        )

        return {
            "commodity": display_name,
            "narrative": narrative,
            "forecast_pct": round(forecast_pct * 100, 1),
            "exposure_gbp": round(exposure_gbp, 0),
            "hedge_ratio": hedge_ratio,
            "llm_backend": llm._backend,
            "source": "live" if fc else "warming",
        }
    except Exception as exc:
        logger.warning(f"/intelligence/narrative/{commodity} error: {exc}")
        return {
            "commodity": commodity,
            "narrative": f"Narrative generation unavailable — {exc}",
            "source": "error",
            "error": str(exc),
        }


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
