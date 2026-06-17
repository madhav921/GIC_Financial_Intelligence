"""
GIC Plan-to-Perform Engine — Main Orchestrator

Ties all 5 layers together in a single, readable entry point.
Each layer is independently testable and replaceable.

Architecture:
  Layer 1: Data Architecture       → DataLayerController
  Layer 2: Predictive Intelligence → IntelligenceLayerController
  Layer 3: Financial Drivers       → FinancialLayerController
  Layer 4: Simulation & Risk       → SimulationLayerController
  Layer 5: Governance & LLM        → GovernanceLayerController

Usage:
  from orchestrator import GICOrchestrator
  engine = GICOrchestrator()
  results = engine.run_full_pipeline()
"""
from __future__ import annotations
import time
from loguru import logger

from layers.layer1_data.controller import DataLayerController
from layers.layer2_intelligence.controller import IntelligenceLayerController
from layers.layer3_financial.controller import FinancialLayerController
from layers.layer4_simulation.controller import SimulationLayerController
from layers.layer5_governance.controller import GovernanceLayerController


class GICOrchestrator:
    """
    GIC Plan-to-Perform — 5-Layer Orchestrator

    Data flow: L1 → L2 → L3 → L4 → L5
      L1 (Data)         → commodity_df, macro_df, sales_df
      L2 (Intelligence) → forecasts, commodity_index
      L3 (Financial)    → monthly P&L, annual summary
      L4 (Simulation)   → Monte Carlo results, scenario comparison
      L5 (Governance)   → audit trail, LLM narratives, bias tracking
    """

    def __init__(self):
        logger.info("GIC Orchestrator: Initializing 5-layer engine")
        self.data         = DataLayerController()          # Layer 1
        self.intelligence = IntelligenceLayerController()  # Layer 2
        self.financial    = FinancialLayerController()     # Layer 3
        self.simulation   = SimulationLayerController()    # Layer 4
        self.governance   = GovernanceLayerController()    # Layer 5
        logger.info("GIC Orchestrator: All 5 layers initialized")

    def run_full_pipeline(self, n_simulations: int = 10_000) -> dict:
        """
        Execute the complete GIC pipeline across all 5 layers.
        Returns a comprehensive results dict with P&L, forecasts, risk metrics, and narratives.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("GIC Orchestrator: Starting full pipeline run")
        logger.info("=" * 60)

        # ── Layer 1: Data Acquisition ─────────────────────────────────────────
        logger.info("LAYER 1 → Data Architecture: Loading datasets")
        commodity_df, macro_df, sales_df, bom_df = self.data.load_all()
        self.governance.log_event("data_loaded", {
            "commodity_rows": len(commodity_df),
            "macro_rows": len(macro_df) if macro_df is not None else 0,
            "sales_rows": len(sales_df),
        })

        # ── Layer 2: Predictive Intelligence ─────────────────────────────────
        logger.info("LAYER 2 → Predictive Intelligence: Training models & generating forecasts")
        training_metrics = self.intelligence.train_all_models(commodity_df, macro_df)
        commodity_index_df = self.intelligence.generate_commodity_index(commodity_df)
        forecasts = self.intelligence.forecast_all_commodities(commodity_df, macro_df)
        self.governance.log_event("models_trained", {
            "n_commodities": len(training_metrics),
            "n_forecasts": len(forecasts),
        })

        # ── Layer 3: Financial Drivers ────────────────────────────────────────
        logger.info("LAYER 3 → Financial Drivers: Building P&L")
        pnl_df = self.financial.build_pnl(sales_df, commodity_index_df)
        annual_df = self.financial.annual_summary(pnl_df)
        self.governance.log_event("pnl_generated", {
            "months": len(pnl_df),
            "total_revenue": float(pnl_df["net_revenue"].sum()),
        })

        # ── Layer 4: Simulation & Risk ────────────────────────────────────────
        logger.info("LAYER 4 → Simulation & Risk: Running Monte Carlo")
        mc_result = self.simulation.run_monte_carlo(
            sales_df, commodity_index_df, n_simulations=n_simulations
        )
        scenario_results = self.simulation.run_all_scenarios(sales_df, commodity_index_df)
        scenario_comparison = self.simulation.compare_scenarios(scenario_results)
        risk_decomp = self.simulation.decompose_risk(sales_df, commodity_index_df)
        self.governance.log_event("simulation_run", {
            "n_simulations": n_simulations,
            "n_scenarios": len(scenario_results),
            "commodity_risk_pct": risk_decomp.get("commodity_pct"),
        })

        # ── Layer 5: Governance & LLM Narratives ─────────────────────────────
        logger.info("LAYER 5 → Governance: Generating LLM narratives & audit trail")
        narratives = {}
        for commodity, forecast in list(forecasts.items())[:3]:  # Top 3 for demo
            try:
                narratives[commodity] = self.governance.generate_narrative(forecast)
            except Exception as e:
                logger.warning(f"Narrative generation failed for {commodity}: {e}")

        # Executive insight
        total_revenue = float(pnl_df["net_revenue"].sum())
        gross_margin = float(pnl_df["gross_margin"].sum())
        executive_insight = self.governance.generate_executive_insight(
            pnl_summary={
                "total_revenue": total_revenue,
                "gross_margin_pct": (
                    (gross_margin / total_revenue * 100) if total_revenue > 0 else 0
                ),
                "ebit": float(pnl_df["operating_income"].sum()),
            },
            commodity_index=(
                float(commodity_index_df["commodity_index"].iloc[-1])
                if "commodity_index" in commodity_index_df.columns
                else 100.0
            ),
        )
        llm_status = self.governance.llm_health_check()

        # ── Compile results ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        logger.info(f"GIC Orchestrator: Pipeline complete in {elapsed:.1f}s")

        self.governance.log_event("pipeline_complete", {"elapsed_seconds": round(elapsed, 1)})

        return {
            "layer1_data": {
                "commodity_shape": commodity_df.shape,
                "macro_shape": macro_df.shape if macro_df is not None else None,
                "sales_shape": sales_df.shape,
            },
            "layer2_intelligence": {
                "training_metrics": training_metrics,
                "n_forecasts": len(forecasts),
                "commodity_index_latest": (
                    float(commodity_index_df["commodity_index"].iloc[-1])
                    if "commodity_index" in commodity_index_df.columns
                    else None
                ),
            },
            "layer3_financial": {
                "total_revenue": total_revenue,
                "total_cogs": float(pnl_df["total_cogs"].sum()),
                "gross_margin": gross_margin,
                "ebit": float(pnl_df["operating_income"].sum()),
                "annual_summary": (
                    annual_df.to_dict("records") if not annual_df.empty else []
                ),
            },
            "layer4_simulation": {
                "mc_stats": mc_result.stats,
                "scenario_comparison": scenario_comparison.to_dict("records"),
                "risk_decomposition": risk_decomp,
            },
            "layer5_governance": {
                "narratives": narratives,
                "executive_insight": executive_insight,
                "llm_status": llm_status,
                "audit_events": len(self.governance.get_audit_history(limit=20)),
            },
            "pipeline_elapsed_seconds": round(elapsed, 1),
        }

    def quick_pnl(self, demand_shock: float = 0.0, commodity_shock: float = 0.0) -> dict:
        """Quick P&L calculation with optional scenario shocks. <5 seconds."""
        commodity_df, macro_df, sales_df, _ = self.data.load_all()
        commodity_index_df = self.intelligence.generate_commodity_index(commodity_df)
        pnl_df = self.financial.apply_scenario(
            sales_df, commodity_index_df, demand_shock, commodity_shock
        )
        return {
            "total_revenue": float(pnl_df["net_revenue"].sum()),
            "gross_margin": float(pnl_df["gross_margin"].sum()),
            "ebit": float(pnl_df["operating_income"].sum()),
            "demand_shock": demand_shock,
            "commodity_shock": commodity_shock,
        }


if __name__ == "__main__":
    engine = GICOrchestrator()
    results = engine.run_full_pipeline()
    import json
    print(json.dumps(
        {
            k: str(v) if not isinstance(v, (dict, list, float, int, str, bool, type(None))) else v
            for k, v in results.items()
        },
        indent=2,
        default=str,
    ))
