"""Simulation & scenario API routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.schemas import ScenarioRequest, ScenarioResponse
from src.data.data_loader import DataLoader
from src.drivers.financial_model import FinancialModel
from src.governance.audit_trail import AuditTrail
from src.models.commodity_forecast import CommodityForecastModel
from src.simulation.monte_carlo import MonteCarloEngine
from src.simulation.scenario_engine import ScenarioDefinition, ScenarioEngine

router = APIRouter(prefix="/simulation", tags=["simulation"])
audit = AuditTrail()

N_HISTOGRAM_BINS = 30


def _build_histogram_bins(oi_dist: np.ndarray, n_bins: int = N_HISTOGRAM_BINS) -> list[dict]:
    """Convert an operating-income distribution array into histogram bins in £M."""
    lo = float(np.min(oi_dist))
    hi = float(np.max(oi_dist))
    width = (hi - lo) / n_bins if hi > lo else 1.0
    counts = [0] * n_bins
    for v in oi_dist:
        idx = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1
    return [
        {"x": round((lo + width * (i + 0.5)) / 1e6, 2), "count": counts[i]}
        for i in range(n_bins)
    ]


@router.post("/scenario", response_model=ScenarioResponse)
async def run_scenario(request: ScenarioRequest):
    """Run a what-if scenario with Monte Carlo simulation."""
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        commodity_df = loader.load_commodity_prices()

        cfm = CommodityForecastModel()
        commodity_index_df = cfm.generate_commodity_index(commodity_df)

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
        histogram_bins = None
        if "simulation" in result:
            sim_result = result["simulation"]
            sim_stats = sim_result.stats
            histogram_bins = _build_histogram_bins(sim_result.operating_income_dist)

        audit.log_scenario_run(
            scenario_name=request.name,
            parameters={"demand_shock": request.demand_shock, "commodity_shock": request.commodity_shock},
            result_summary=det_summary,
        )

        return ScenarioResponse(
            scenario_name=request.name,
            deterministic=det_summary,
            simulation_stats=sim_stats,
            histogram_bins=histogram_bins,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/variance-decomposition")
async def variance_decomposition():
    """
    Decompose P&L uncertainty by risk source: commodity, demand, FX.

    Runs 3 partial Monte Carlo simulations (holding 2 sources fixed each time)
    then attributes variance proportionally. Returns percentage contributions
    and absolute risk metrics (VaR95, CVaR95) in £.
    """
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        commodity_df = loader.load_commodity_prices()

        cfm = CommodityForecastModel()
        commodity_index_df = cfm.generate_commodity_index(commodity_df)

        mc_engine = MonteCarloEngine()
        result = mc_engine.decompose_variance(
            sales_df=sales_df,
            commodity_index_df=commodity_index_df,
            n_simulations=3000,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monthly-fan")
async def monthly_fan():
    """
    Run a monthly Monte Carlo fan chart for operating income.

    Returns per-month percentile bands (p5/p10/p25/p75/p90/p95) and mean,
    all in £M, suitable for the FanChart component.
    """
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        commodity_df = loader.load_commodity_prices()

        cfm = CommodityForecastModel()
        commodity_index_df = cfm.generate_commodity_index(commodity_df)

        financial_model = FinancialModel()
        base_pnl = financial_model.build_pnl(sales_df, commodity_index_df)

        mc_engine = MonteCarloEngine()
        fan_df = mc_engine.run_monthly_fan(base_pnl, n_simulations=2000)

        scale = 1e6  # Convert raw £ → £M for frontend
        records = []
        for _, row in fan_df.iterrows():
            date_val = row["date"]
            date_str = str(date_val)[:7] if hasattr(date_val, "__str__") else str(date_val)
            records.append({
                "date": date_str,
                "mean": round(float(row["mean_oi"]) / scale, 2),
                "p5":   round(float(row["p5_oi"])   / scale, 2),
                "p10":  round(float(row["p10_oi"])  / scale, 2),
                "p25":  round(float(row["p25_oi"])  / scale, 2),
                "p75":  round(float(row["p75_oi"])  / scale, 2),
                "p90":  round(float(row["p90_oi"])  / scale, 2),
                "p95":  round(float(row["p95_oi"])  / scale, 2),
            })
        return {"months": records}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
