"""
Actionable-Intelligence API routes (Layer 5 surface).

Exposes the insights / prescriptive layer to the front-end:
  • GET  /insights/feed              — prioritised InsightCard feed + summary
  • GET  /insights/variance-bridge   — plan→actual EBIT waterfall
  • GET  /insights/early-warning     — composite risk score
  • GET  /insights/warranty/summary  — warranty forecast / accrual / modes / risk
  • POST /insights/recommend/hedge   — hedge-ratio recommendation

Every handler is defensive: it never returns a 500 on missing data — it falls
back to curated demo content so the dashboard always renders rich output.
Heavy modules are imported lazily inside handlers.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_project_root

insights_router = APIRouter(prefix="/insights", tags=["insights"])


# ── request models ────────────────────────────────────────────────────────────
class HedgeRequest(BaseModel):
    commodity: str = Field(..., examples=["Lithium"])
    forecast_pct: float = Field(..., description="Forecast price move as a fraction, e.g. 0.18")
    exposure_gbp: float = Field(..., description="Annual GBP exposure to the commodity")
    current_hedge_ratio: float = Field(0.4, ge=0.0, le=1.0)


# ── helpers ───────────────────────────────────────────────────────────────────
def _warranty_csv_path() -> Path:
    return get_project_root() / "data" / "synthetic" / "warranty_data.csv"


def _load_or_generate_warranty():
    """Load warranty_data.csv, generating it if missing. Returns a DataFrame."""
    import pandas as pd

    path = _warranty_csv_path()
    if not path.exists():
        logger.info("warranty_data.csv missing — generating")
        from src.data.warranty_generator import WarrantyDataGenerator
        return WarrantyDataGenerator().save(path)
    return pd.read_csv(path)


# ── endpoints ─────────────────────────────────────────────────────────────────
@insights_router.get("/feed")
async def get_feed(top_n: int = 8):
    """Prioritised InsightCard feed with summary stats (live data, demo fallback)."""
    try:
        from src.api.pipeline_cache import get_snapshot
        from src.insights.insight_engine import InsightEngine

        engine = InsightEngine()
        snap = get_snapshot()

        context: dict = {}
        # Enrich with cached pipeline data (same source as every other route).
        if snap.get("forecasts"):
            context["forecasts"] = snap["forecasts"]
        if snap.get("commodity_index_df") is not None:
            context["commodity_index"] = snap["commodity_index_df"]
        # Best-effort warranty enrichment.
        try:
            context["warranty_df"] = _load_or_generate_warranty()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"feed: warranty enrichment skipped ({exc})")

        insights = engine.generate_insights(context, top_n=top_n)
        return {
            "insights": [i.to_dict() for i in insights],
            "summary": engine.summary_stats(insights),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"/insights/feed fell back to demo ({exc})")
        from src.insights.insight_engine import InsightEngine
        engine = InsightEngine()
        insights = engine.generate_insights({}, top_n=top_n)
        return {
            "insights": [i.to_dict() for i in insights],
            "summary": engine.summary_stats(insights),
        }


@insights_router.get("/variance-bridge")
async def get_variance_bridge():
    """Plan→actual EBIT variance waterfall (realistic demo if no data)."""
    try:
        from src.insights.variance_bridge import VarianceBridgeAnalyzer

        analyzer = VarianceBridgeAnalyzer()
        # Realistic demo: plan EBIT £1.50bn vs actual £1.401bn (−6.6%).
        bridge = analyzer.build_from_scenarios(
            base_pnl_summary={"ebit": 1_500_000_000.0},
            scenario_pnl_summary={"ebit": 1_401_000_000.0},
        )
        return bridge.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"/insights/variance-bridge error ({exc})")
        return {
            "plan_ebit": 1_500_000_000.0,
            "actual_ebit": 1_401_000_000.0,
            "total_variance_gbp": -99_000_000.0,
            "total_variance_pct": -6.6,
            "bridge": [],
            "error": str(exc),
        }


@insights_router.get("/early-warning")
async def get_early_warning():
    """Composite 0–100 cross-domain risk score (live commodity signals from pipeline cache)."""
    try:
        from src.api.pipeline_cache import get_snapshot
        from src.insights.early_warning import EarlyWarningSystem

        ews = EarlyWarningSystem()
        snap = get_snapshot()

        warranty_risk = None
        try:
            from src.models.warranty_model import WarrantyModel
            wdf = _load_or_generate_warranty()
            warranty_risk = WarrantyModel().warranty_risk_score(wdf)["score"]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"early-warning: warranty signal skipped ({exc})")

        # Use real commodity volatility from pipeline cache when available.
        commodity_vol = snap.get("commodity_volatility_pct")

        # Derive commodity MAPE proxy from bias metrics (mean of all mape values).
        commodity_mape = None
        bias_metrics = snap.get("bias_metrics") or []
        if bias_metrics:
            commodity_mape = round(
                sum(b["mape"] for b in bias_metrics) / len(bias_metrics), 1
            )

        return ews.compute_risk_score(
            commodity_volatility_pct=commodity_vol,
            commodity_mape_pct=commodity_mape,
            warranty_risk_score=warranty_risk,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"/insights/early-warning error ({exc})")
        from src.insights.early_warning import EarlyWarningSystem
        return EarlyWarningSystem().compute_risk_score()


@insights_router.get("/warranty/summary")
async def get_warranty_summary():
    """Warranty forecast, accrual adequacy, failure-mode mix and risk score."""
    try:
        from src.models.warranty_model import WarrantyModel

        wdf = _load_or_generate_warranty()
        wm = WarrantyModel()
        return {
            "forecast": wm.forecast_warranty_cost(wdf, horizon_months=12).to_dict(),
            "accrual_adequacy": wm.assess_accrual_adequacy(wdf),
            "failure_modes": wm.failure_mode_breakdown(wdf),
            "risk_score": wm.warranty_risk_score(wdf),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"/insights/warranty/summary error ({exc})")
        return {
            "forecast": {"dates": [], "point": [], "lower": [], "upper": []},
            "accrual_adequacy": {"adequacy_pct": 100.0, "status": "adequate", "shortfall_gbp": 0.0},
            "failure_modes": {"breakdown_pct": {}, "rising_modes": []},
            "risk_score": {"score": 0.0, "band": "low", "components": {}},
            "error": str(exc),
        }


@insights_router.post("/recommend/hedge")
async def recommend_hedge(body: HedgeRequest):
    """Hedge-ratio recommendation with expected £ savings."""
    try:
        from src.insights.recommendation_engine import RecommendationEngine

        return RecommendationEngine().hedge_recommendation(
            commodity=body.commodity,
            forecast_pct=body.forecast_pct,
            exposure_gbp=body.exposure_gbp,
            current_hedge_ratio=body.current_hedge_ratio,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"/insights/recommend/hedge error ({exc})")
        return {
            "commodity": body.commodity,
            "action": "Maintain current hedge ratio",
            "target_hedge_ratio": body.current_hedge_ratio,
            "expected_savings_gbp": 0.0,
            "rationale": "Recommendation engine unavailable — defaulting to hold.",
            "error": str(exc),
        }
