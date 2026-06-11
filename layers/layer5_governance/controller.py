"""
Layer 5: Governance & Explainability Controller

Single entry point for audit logging, LLM narrative generation,
bias tracking, and model explainability.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
from loguru import logger

from src.governance.audit_trail import AuditTrail
from src.governance.bias_tracking import BiasTracker
from src.governance.explainability import ExplainabilityEngine
from layers.layer5_governance.llm_engine import GICLLMEngine, LLMConfig
from src.models.commodity_forecast import ForecastResult


class GovernanceLayerController:
    """
    Layer 5 Controller — Governance & Explainability

    Usage:
        gov = GovernanceLayerController()
        event_id = gov.log_event("forecast_generated", {"commodity": "Copper", "mape": 7.0})
        narrative = gov.generate_narrative(forecast_result)
        explanation = gov.explain_forecast(commodity, forecast_result)
        bias_report = gov.track_bias(commodity, actual_prices, forecast_prices)
        llm_status = gov.llm_health_check()
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self._audit = AuditTrail()
        self._bias_tracker = BiasTracker()
        self._explainability = ExplainabilityEngine()
        self._llm = GICLLMEngine(llm_config)
        logger.info("Layer 5: Governance controller initialized")

    # ── Audit Trail ───────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        details: dict,
        user: str = "system",
    ) -> str:
        """
        Append an immutable event to the JSONL audit trail.
        Returns the event UUID for traceability.
        """
        return self._audit.log_event(event_type, details, user=user)

    def get_audit_history(self, limit: int = 100) -> list[dict]:
        """Return recent audit trail events."""
        return self._audit.get_recent_events(limit=limit)

    # ── LLM Narrative Generation ──────────────────────────────────────────────

    def generate_narrative(self, forecast_result: ForecastResult) -> str:
        """
        Generate an LLM-powered natural language narrative for a commodity forecast.
        Uses open-source flan-t5 or Ollama backend.
        """
        pct_change = 0.0
        if forecast_result.point_forecast and len(forecast_result.point_forecast) > 1:
            first = forecast_result.point_forecast[0]
            last = forecast_result.point_forecast[-1]
            pct_change = ((last - first) / (first + 1e-9)) * 100

        # Get top feature drivers if available
        drivers = (
            list(forecast_result.feature_importance.keys())[:3]
            if forecast_result.feature_importance
            else ["macro conditions", "historical trend", "supply-demand"]
        )

        narrative = self._llm.explain_forecast(
            commodity=forecast_result.commodity,
            forecast_pct=pct_change,
            drivers=drivers,
            mape=forecast_result.metrics.get("cv_mape_mean"),
        )

        # Log narrative generation
        self.log_event("narrative_generated", {
            "commodity": forecast_result.commodity,
            "model_type": forecast_result.model_type,
            "llm_backend": self._llm._backend,
            "forecast_pct": round(pct_change, 2),
        })

        return narrative

    def generate_risk_narrative(
        self,
        scenario_name: str,
        simulation_result,
        risk_decomposition: dict,
    ) -> str:
        """Generate LLM-powered risk scenario narrative."""
        stats = simulation_result.stats.get("operating_income", {})
        mean_oi = stats.get("mean", 0)
        var_95 = stats.get("var_95", 0)

        # Compute EBIT impact vs base (placeholder — base comparison would require base result)
        ebit_impact_pct = 0.0

        return self._llm.generate_risk_summary(
            scenario_name=scenario_name,
            ebit_impact_pct=ebit_impact_pct,
            var_95=var_95,
            commodity_risk_pct=risk_decomposition.get("commodity_pct", 60),
            demand_risk_pct=risk_decomposition.get("demand_pct", 30),
            fx_risk_pct=risk_decomposition.get("fx_pct", 10),
        )

    def generate_executive_insight(
        self,
        pnl_summary: dict,
        commodity_index: float = 100.0,
        top_risk_commodity: str = "Lithium",
    ) -> str:
        """Generate dashboard executive insight using LLM."""
        return self._llm.generate_executive_insight(
            total_revenue=pnl_summary.get("total_revenue", 0),
            gross_margin_pct=pnl_summary.get("gross_margin_pct", 0),
            ebit=pnl_summary.get("ebit", 0),
            commodity_index=commodity_index,
            top_risk_commodity=top_risk_commodity,
        )

    # ── Bias Tracking ─────────────────────────────────────────────────────────

    def track_bias(
        self,
        commodity: str,
        actual_prices: list[float],
        forecast_prices: list[float],
    ) -> dict:
        """
        Track forecast bias for a commodity.
        >5% bias → alert | >10% bias → governance escalation.
        Returns: {bias_pct, mae, mape, alert_triggered, escalation_required}
        """
        result = self._bias_tracker.compute_bias(commodity, actual_prices, forecast_prices)

        if result.get("escalation_required"):
            alert_narrative = self._llm.explain_alert(
                commodity=commodity,
                alert_type="bias_escalation",
                variance_pct=result["bias_pct"],
                threshold_pct=10.0,
            )
            self.log_event("bias_escalation", {
                "commodity": commodity,
                "bias_pct": result["bias_pct"],
                "narrative": alert_narrative,
            })
        elif result.get("alert_triggered"):
            self.log_event("bias_alert", {
                "commodity": commodity,
                "bias_pct": result["bias_pct"],
            })

        return result

    # ── Explainability ────────────────────────────────────────────────────────

    def explain_forecast(
        self,
        commodity: str,
        forecast_result: ForecastResult,
    ) -> dict:
        """
        Generate feature importance explanation for a commodity forecast.
        Returns: {commodity, top_drivers, narrative, forecast_value, trend}
        """
        explanation = self._explainability.explain(commodity, forecast_result)
        return {
            "commodity": commodity,
            "top_drivers": (
                explanation.top_drivers if hasattr(explanation, "top_drivers") else []
            ),
            "narrative": (
                explanation.narrative if hasattr(explanation, "narrative") else ""
            ),
            "llm_narrative": self.generate_narrative(forecast_result),
            "forecast_value": (
                explanation.forecast_value if hasattr(explanation, "forecast_value") else 0
            ),
        }

    # ── LLM Health ────────────────────────────────────────────────────────────

    def llm_health_check(self) -> dict:
        """Return LLM backend status."""
        return self._llm.health_check()
