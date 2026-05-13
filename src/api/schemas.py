"""API schemas (Pydantic models)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: int


class CommodityForecastRequest(BaseModel):
    commodity: str = Field(..., description="Commodity name (e.g., Lithium, Steel)")
    horizon_months: int = Field(12, ge=1, le=36)


class CommodityForecastResponse(BaseModel):
    commodity: str
    model_type: str
    dates: list[str]
    point_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]
    metrics: dict[str, float]


class ScenarioRequest(BaseModel):
    name: str = Field("custom", description="Scenario name")
    demand_shock: float = Field(0.0, ge=-1.0, le=1.0, description="Demand shock (-1 to 1)")
    commodity_shock: float = Field(0.0, ge=-1.0, le=2.0, description="Commodity shock")
    fx_shock: float = Field(0.0, ge=-1.0, le=1.0, description="FX shock")
    n_simulations: int = Field(5000, ge=100, le=100000)


class ScenarioResponse(BaseModel):
    scenario_name: str
    deterministic: dict
    simulation_stats: dict | None = None


class FinancialSummaryResponse(BaseModel):
    period: str
    segments: list[dict]
    total_revenue: float
    total_margin: float
    margin_pct: float


class ElasticityResponse(BaseModel):
    segment: str
    own_price_elasticity: float
    incentive_elasticity: float
    commodity_cross_elasticity: float
    r_squared: float


class ShockRequest(BaseModel):
    """Request body for POST /pnl/shock."""
    shocks: dict[str, float] = Field(
        ...,
        description="Commodity name → fractional shock (e.g. {\"Lithium\": 0.20 means +20%})",
    )
    base_revenue: float | None = Field(
        None,
        description="Override base annual revenue (USD). Defaults to latest P&L sum.",
    )


class ShockWaterfallItem(BaseModel):
    commodity: str
    shock_pct: float
    cogs_impact: float
    ebit_impact: float
    margin_impact_bps: float
    pct_of_base_ebit: float


class ShockResponse(BaseModel):
    waterfall: list[ShockWaterfallItem]
    total_ebit_impact: float
    total_cogs_impact: float
    total_margin_impact_bps: float
    base_revenue: float
    hedge_recommendations: list[dict]


class RegimeInfo(BaseModel):
    commodity: str
    regime: str
    hurst: float
    rolling_vol_pct: float
    ensemble_weights: dict[str, float]
    confidence: str
