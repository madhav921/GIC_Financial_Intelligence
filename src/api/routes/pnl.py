"""
P&L Shock API routes.

POST /pnl/shock   — compute BOM-weighted P&L waterfall for commodity price shocks
GET  /pnl/regime  — return current Hurst-based regime for all tracked commodities
"""

from __future__ import annotations

import logging
import math

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import RegimeInfo, ShockRequest, ShockResponse, ShockWaterfallItem
from src.config import get_settings
from src.models.commodity_shock import CommodityShockCalculator
from src.models.hedge_optimizer import HedgeOptimizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pnl", tags=["P&L"])


@router.get("/annual", summary="Annual P&L summary for the dashboard KPI strip")
def get_annual_pnl():
    """
    Return aggregated annual P&L KPIs used by the Executive Summary dashboard.
    Builds from synthetic sales + commodity index; falls back to config defaults.
    """
    try:
        from layers.layer1_data.controller import DataLayerController
        from layers.layer2_intelligence.controller import IntelligenceLayerController
        from layers.layer3_financial.controller import FinancialLayerController

        data = DataLayerController()
        intel = IntelligenceLayerController()
        fin = FinancialLayerController()

        commodity_df, _, sales_df, _ = data.load_all()
        commodity_index_df = intel.generate_commodity_index(commodity_df)
        pnl_df = fin.build_pnl(sales_df, commodity_index_df)
        annual_df = fin.annual_summary(pnl_df)

        # Use the most recent calendar year for the annual KPI strip.
        # annual_df is already aggregated by (year, segment); sum all segments for the latest year.
        most_recent_year = int(annual_df["year"].max()) if not annual_df.empty else None
        if most_recent_year and not annual_df.empty:
            yr = annual_df[annual_df["year"] == most_recent_year]
            total_rev = float(yr["net_revenue"].sum())
            gross_margin = float(yr["gross_margin"].sum())
            ebit = float(yr["operating_income"].sum())
        else:
            # Fallback: group pnl_df by the most recent calendar year
            import pandas as _pd
            pnl_df["_year"] = _pd.to_datetime(pnl_df["date"]).dt.year
            yr_df = pnl_df[pnl_df["_year"] == pnl_df["_year"].max()]
            total_rev = float(yr_df["net_revenue"].sum())
            gross_margin = float(yr_df["gross_margin"].sum())
            ebit = float(yr_df["operating_income"].sum())

        # Raise if any core metric is NaN/inf/zero — triggers config-based fallback below.
        if not all(math.isfinite(v) for v in [total_rev, gross_margin, ebit]):
            raise ValueError(f"NaN/inf in P&L: rev={total_rev} gm={gross_margin} ebit={ebit}")
        if total_rev <= 0:
            raise ValueError(f"Non-positive revenue ({total_rev}); cannot compute P&L ratios")

        # Segment breakdown — most-recent calendar year only, consistent with KPI strip
        segments = []
        if "segment" in sales_df.columns:
            sf = sales_df.copy()
            if "date" in sf.columns:
                sf["_year"] = pd.to_datetime(sf["date"]).dt.year
                sf = sf[sf["_year"] == int(sf["_year"].max())]
            price_col = next(
                (c for c in ("avg_price_usd", "price", "unit_price") if c in sf.columns),
                None,
            )
            if price_col:
                sf = sf.copy()
                sf["_rev"] = sf["volume"] * sf[price_col]
                seg_agg = sf.groupby("segment")[["_rev", "volume"]].sum().reset_index()
                seg_agg = seg_agg.rename(columns={"_rev": "revenue"})
            else:
                seg_agg = sf.groupby("segment")[["volume"]].sum().reset_index()
                seg_agg["revenue"] = 0.0
            segments = [
                {
                    "segment": str(row["segment"]),
                    "revenue": float(row["revenue"]),
                    "volume": int(row["volume"]),
                }
                for _, row in seg_agg.iterrows()
            ]

        return {
            "total_revenue": round(total_rev, 0),
            "gross_margin_pct": round(gross_margin / total_rev * 100 if total_rev else 0, 1),
            "ebit": round(ebit, 0),
            "net_income": round(ebit * 0.79, 0),  # approx 21% tax
            "annual_rows": annual_df.to_dict("records") if not annual_df.empty else [],
            "segments": segments,
        }
    except Exception as exc:
        logger.warning(f"/pnl/annual fallback: {exc}")
        # Config-based fallback so the dashboard always renders
        settings = get_settings()
        base_rev = sum(
            s.get("avg_price_usd", 0) * s.get("annual_volume", 0)
            for s in settings.get("vehicle_segments", [])
        ) or 19_800_000_000
        return {
            "total_revenue": base_rev,
            "gross_margin_pct": 18.5,
            "ebit": base_rev * 0.071,
            "net_income": base_rev * 0.056,
            "annual_rows": [],
            "segments": [],
        }


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
