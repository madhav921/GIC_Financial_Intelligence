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
