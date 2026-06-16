"""Forecast API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import CommodityForecastRequest, CommodityForecastResponse, ElasticityResponse
from src.data.data_loader import DataLoader
from src.governance.audit_trail import AuditTrail
from src.models.commodity_forecast import CommodityForecastModel
from src.models.price_elasticity import PriceElasticityModel

router = APIRouter(prefix="/forecast", tags=["forecast"])
audit = AuditTrail()


@router.post("/commodity", response_model=CommodityForecastResponse)
async def forecast_commodity(request: CommodityForecastRequest):
    """Generate a commodity price forecast."""
    try:
        loader = DataLoader()
        commodity_df = loader.load_commodity_prices()
        macro_df = loader.load_macro_indicators()

        if request.commodity not in commodity_df.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Commodity '{request.commodity}' not found. "
                       f"Available: {[c for c in commodity_df.columns if c != 'date']}",
            )

        model = CommodityForecastModel()
        model.settings["forecast"]["horizon_months"] = request.horizon_months

        # Set DatetimeIndex so SARIMAX gets correct seasonal ordering
        price_series = commodity_df.set_index("date")[request.commodity]
        metrics = model.train_sarimax(request.commodity, price_series)
        result = model.forecast_sarimax(request.commodity)

        # Audit log
        audit.log_forecast(
            model_name="sarimax",
            commodity=request.commodity,
            forecast_values=result.point_forecast,
            metrics=metrics,
        )

        # Generate JLR CFO narrative using pipeline cache data for exposure context.
        narrative = None
        try:
            from src.api.pipeline_cache import get_snapshot
            from layers.layer5_governance.llm_engine import GICLLMEngine, _COMMODITY_CONTEXT
            snap = get_snapshot()
            fc_data = (snap.get("forecasts") or {}).get(request.commodity, {})
            ctx = _COMMODITY_CONTEXT.get(request.commodity, {})
            drivers = ctx.get("drivers", list((result.feature_importance or {}).keys())[:3])
            narrative = GICLLMEngine().explain_forecast(
                commodity=request.commodity,
                forecast_pct=fc_data.get("forecast_pct", 0.0),
                drivers=drivers,
                exposure_gbp=fc_data.get("exposure_gbp", 0.0),
                hedge_ratio=fc_data.get("hedge_ratio", 0.40),
                mape=metrics.get("cv_mape_mean") or metrics.get("mape"),
            )
        except Exception as _narr_exc:
            import logging as _l
            _l.getLogger(__name__).debug(f"Narrative generation skipped: {_narr_exc}")

        return CommodityForecastResponse(
            commodity=result.commodity,
            model_type=result.model_type,
            dates=result.dates,
            point_forecast=result.point_forecast,
            lower_80=result.lower_80,
            upper_80=result.upper_80,
            lower_95=result.lower_95,
            upper_95=result.upper_95,
            metrics=metrics,
            narrative=narrative,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commodity-index")
async def get_commodity_index():
    """Get the computed commodity index."""
    try:
        loader = DataLoader()
        commodity_df = loader.load_commodity_prices()
        model = CommodityForecastModel()
        index_df = model.generate_commodity_index(commodity_df)
        return {"data": index_df.to_dict(orient="records")}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/commodity-history")
async def get_commodity_history(commodity: str = "Steel", months: int = 36):
    """Return historical commodity price series from the real data source.

    Used by the Commodity Intelligence chart to show real price history before
    the forecast horizon — ensures the history and forecast are on the same scale.

    Args:
        commodity: commodity name matching the CSV column (e.g. Steel, Lithium)
        months:    number of historical months to return (default 36)

    Returns:
        {"commodity": str, "unit": str, "data_source": str,
         "history": [{"date": "YYYY-MM", "price": float}]}
    """
    try:
        loader = DataLoader()
        commodity_df = loader.load_commodity_prices()

        if commodity not in commodity_df.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Commodity '{commodity}' not found. "
                       f"Available: {[c for c in commodity_df.columns if c != 'date']}",
            )

        df = commodity_df[["date", commodity]].dropna()
        df = df.sort_values("date").tail(months)

        history = [
            {"date": str(row["date"])[:7], "price": round(float(row[commodity]), 4)}
            for _, row in df.iterrows()
        ]

        data_source = loader.get_data_source("commodity_prices")

        return {
            "commodity": commodity,
            "data_source": data_source,
            "history": history,
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/elasticity")
async def get_price_elasticity():
    """Get price elasticity estimates for all segments."""
    try:
        loader = DataLoader()
        sales_df = loader.load_sales_data()
        macro_df = loader.load_macro_indicators()
        commodity_df = loader.load_commodity_prices()

        model = PriceElasticityModel()
        model.fit_all_segments(sales_df, macro_df, commodity_df)
        return {"data": model.summary_table().to_dict(orient="records")}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
