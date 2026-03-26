"""Simulation & scenario API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ScenarioRequest, ScenarioResponse
from src.data.data_loader import DataLoader
from src.governance.audit_trail import AuditTrail
from src.models.commodity_forecast import CommodityForecastModel
from src.simulation.scenario_engine import ScenarioDefinition, ScenarioEngine

router = APIRouter(prefix="/simulation", tags=["simulation"])
audit = AuditTrail()


@router.post("/scenario", response_model=ScenarioResponse)
async def run_scenario(request: ScenarioRequest):
    """Run a what-if scenario with Monte Carlo simulation."""
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        commodity_df = loader.load_commodity_prices()

        # Generate commodity index
        cfm = CommodityForecastModel()
        commodity_index_df = cfm.generate_commodity_index(commodity_df)

        # Define scenario
        scenario = ScenarioDefinition(
            name=request.name,
            description=f"API scenario: demand={request.demand_shock:+.0%}, commodity={request.commodity_shock:+.0%}",
            demand_shock=request.demand_shock,
            commodity_shock=request.commodity_shock,
            fx_shock=request.fx_shock,
        )

        engine = ScenarioEngine()
        result = engine.run_scenario(
            scenario=scenario,
            sales_df=sales_df,
            commodity_index_df=commodity_index_df,
            n_simulations=request.n_simulations,
        )

        # Build response
        det_pnl = result["deterministic_pnl"]
        det_summary = {
            "total_revenue": float(det_pnl["net_revenue"].sum()),
            "total_cogs": float(det_pnl["total_cogs"].sum()),
            "total_margin": float(det_pnl["gross_margin"].sum()),
            "margin_pct": float(
                det_pnl["gross_margin"].sum() / det_pnl["net_revenue"].sum() * 100
                if det_pnl["net_revenue"].sum() != 0 else 0
            ),
            "operating_income": float(det_pnl["operating_income"].sum()),
        }

        sim_stats = None
        if "simulation" in result:
            sim_stats = result["simulation"].stats

        # Audit
        audit.log_scenario_run(
            scenario_name=request.name,
            parameters={"demand_shock": request.demand_shock, "commodity_shock": request.commodity_shock},
            result_summary=det_summary,
        )

        return ScenarioResponse(
            scenario_name=request.name,
            deterministic=det_summary,
            simulation_stats=sim_stats,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets")
async def list_preset_scenarios():
    """List available preset scenarios."""
    presets = ScenarioEngine.preset_scenarios()
    return {
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "demand_shock": s.demand_shock,
                "commodity_shock": s.commodity_shock,
                "fx_shock": s.fx_shock,
            }
            for s in presets
        ]
    }


@router.get("/compare-presets")
async def compare_presets():
    """Run and compare all preset scenarios."""
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        commodity_df = loader.load_commodity_prices()

        cfm = CommodityForecastModel()
        commodity_index_df = cfm.generate_commodity_index(commodity_df)

        engine = ScenarioEngine()
        presets = ScenarioEngine.preset_scenarios()
        comparison = engine.compare_scenarios(presets, sales_df, commodity_index_df)

        return {"data": comparison.to_dict(orient="records")}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
