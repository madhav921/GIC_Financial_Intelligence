"""
P&L Shock API routes.

POST /pnl/shock   — compute BOM-weighted P&L waterfall for commodity price shocks
GET  /pnl/regime  — return current Hurst-based regime for all tracked commodities
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import RegimeInfo, ShockRequest, ShockResponse, ShockWaterfallItem
from src.config import get_settings
from src.models.commodity_shock import CommodityShockCalculator
from src.models.hedge_optimizer import HedgeOptimizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl", tags=["P&L"])


def _get_base_revenue() -> float:
    """Derive base annual revenue from the financial model."""
    try:
        from src.data.data_router import get_operational_source
        from src.drivers.financial_model import FinancialModel

        source = get_operational_source()
        sales_df = source.get_sales("2020-01-01", "2024-12-31")
        pnl = FinancialModel().build_pnl(
            sales_df=sales_df,
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )
        return float(pnl["net_revenue"].sum())
    except Exception:
        settings = get_settings()
        return sum(s["avg_price_usd"] * s["annual_volume"] for s in settings["vehicle_segments"])


@router.post("/shock", response_model=ShockResponse, summary="Compute commodity shock P&L waterfall")
def compute_pnl_shock(request: ShockRequest) -> ShockResponse:
    """
    Simulate the full P&L waterfall impact of simultaneous commodity price shocks.

    Each shock value is a fractional change (e.g. `0.20` = +20%, `-0.15` = -15%).

    Returns:
    - `waterfall`: per-commodity COGS & EBIT impacts, sorted largest-to-smallest absolute impact
    - `total_ebit_impact`: sum of all after-tax EBIT effects (USD)
    - `hedge_recommendations`: optimal hedge ratio for each impacted commodity
    """
    try:
        calc = CommodityShockCalculator()
        settings = get_settings()
        base_revenue = request.base_revenue or _get_base_revenue()

        waterfall_raw = calc.waterfall(request.shocks, base_revenue)

        waterfall = [
            ShockWaterfallItem(
                commodity=row["commodity"],
                shock_pct=row["shock_pct"],
                cogs_impact=row["cogs_impact"],
                ebit_impact=row["ebit_impact"],
                margin_impact_bps=row["margin_impact_bps"],
                pct_of_base_ebit=row["pct_of_base_ebit"],
            )
            for row in waterfall_raw
        ]

        total_ebit = sum(r.ebit_impact for r in waterfall)
        total_cogs = sum(r.cogs_impact for r in waterfall)
        total_bps = sum(r.margin_impact_bps for r in waterfall)

        # Hedge recommendations for top 3 most impacted commodities
        optimizer = HedgeOptimizer()
        hedge_recs = []
        for row in waterfall[:3]:
            key = calc._resolve_key(row.commodity)
            bom_weight = calc.bom_weights.get(key, 0.01)
            material_exposure = (
                base_revenue
                * settings["financial"]["base_cogs_pct"]
                * settings["financial"]["material_cogs_fraction"]
                * bom_weight
            )
            try:
                shock_abs = abs(row.shock_pct)
                hedge = optimizer.optimize(
                    forecast_mean=1.0,
                    forecast_std=max(shock_abs * 0.5, 0.05),
                    futures_price=1.0 * (1 + row.shock_pct * 0.5),
                    exposure_units=material_exposure,
                )
                hedge_recs.append({
                    "commodity": row.commodity,
                    "optimal_hedge_ratio": round(hedge["optimal_hedge_ratio"], 4),
                    "expected_savings_usd": round(hedge["expected_savings"], 0),
                    "var_reduction_usd": round(hedge["var_reduction"], 0),
                    "recommendation": hedge["recommendation"],
                })
            except Exception as exc:
                logger.warning(f"Hedge optimization failed for {row.commodity}: {exc}")

        return ShockResponse(
            waterfall=waterfall,
            total_ebit_impact=round(total_ebit, 0),
            total_cogs_impact=round(total_cogs, 0),
            total_margin_impact_bps=round(total_bps, 1),
            base_revenue=round(base_revenue, 0),
            hedge_recommendations=hedge_recs,
        )

    except Exception as exc:
        logger.exception("Error in /pnl/shock")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/regime", response_model=list[RegimeInfo], summary="Get current market regime for all commodities")
def get_regime() -> list[RegimeInfo]:
    """
    Run Hurst exponent analysis on the most recent 36-month price history
    for each tracked commodity and return regime classification.

    Regime values: `mean_reverting`, `trending`, `volatile`
    """
    try:
        import numpy as np
        from src.data.data_router import get_market_source
        from src.models.regime_detector import RegimeDetector

        settings = get_settings()
        source = get_market_source()
        prices_df = source.get_commodity_prices("2021-01-01", "2024-12-31")

        detector = RegimeDetector()
        results: list[RegimeInfo] = []

        for commodity in settings["commodities"]:
            name = commodity["name"]
            col_candidates = [name, commodity.get("yfinance_ticker", "")]
            col = next((c for c in col_candidates if c in prices_df.columns), None)
            if col is None:
                continue

            series = prices_df[col].dropna().values
            if len(series) < 12:
                continue

            result = detector.detect(series)
            results.append(RegimeInfo(
                commodity=name,
                regime=result["regime"].value,
                hurst=round(float(result["hurst"]), 4),
                rolling_vol_pct=round(float(result["rolling_vol_pct"]), 2),
                ensemble_weights=result["ensemble_weights"],
                confidence=result["confidence"],
            ))

        return results

    except Exception as exc:
        logger.exception("Error in /pnl/regime")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
