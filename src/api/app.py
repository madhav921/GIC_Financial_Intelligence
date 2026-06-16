"""
FastAPI Application — GIC Plan-to-Perform Engine

Serves the AI-powered financial intelligence platform via REST API.
Integrates all layers of the architecture.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.logging_setup import setup_logging

# All route imports are defensive so the API boots even if one module
# fails in the serverless environment (e.g. read-only FS, missing deps).
_IMPORT_ERRORS: dict[str, str] = {}

try:
    from src.api.routes import forecast
except Exception as _e:
    forecast = None  # type: ignore[assignment]
    _IMPORT_ERRORS["forecast"] = repr(_e)

try:
    from src.api.routes import health
except Exception as _e:
    health = None  # type: ignore[assignment]
    _IMPORT_ERRORS["health"] = repr(_e)

try:
    from src.api.routes import pnl
except Exception as _e:
    pnl = None  # type: ignore[assignment]
    _IMPORT_ERRORS["pnl"] = repr(_e)

try:
    from src.api.routes import simulation
except Exception as _e:
    simulation = None  # type: ignore[assignment]
    _IMPORT_ERRORS["simulation"] = repr(_e)

try:
    from src.api.routes.auth import auth_router
except Exception as _e:
    auth_router = None
    _IMPORT_ERRORS["auth"] = repr(_e)

try:
    from src.api.routes.insights import insights_router
except Exception as _e:
    insights_router = None
    _IMPORT_ERRORS["insights"] = repr(_e)

try:
    from src.api.routes.realtime import realtime_router
except Exception as _e:
    realtime_router = None
    _IMPORT_ERRORS["realtime"] = repr(_e)

try:
    from src.api.routes.intelligence import intelligence_router
except Exception as _e:
    intelligence_router = None
    _IMPORT_ERRORS["intelligence"] = repr(_e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="GIC Plan-to-Perform Engine",
        description=(
            "AI-Powered Financial Intelligence: Commodity Forecasting, "
            "Demand Prediction, Scenario Simulation & Driver-Based Financial Modelling"
        ),
        version=settings["project"]["version"],
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes — skip any that failed to import
    if health is not None:
        app.include_router(health.router)
    if forecast is not None:
        app.include_router(forecast.router)
    if simulation is not None:
        app.include_router(simulation.router)
    if pnl is not None:
        app.include_router(pnl.router)
    if auth_router is not None:
        app.include_router(auth_router)
    if insights_router is not None:
        app.include_router(insights_router)
    if realtime_router is not None:
        app.include_router(realtime_router)
    if intelligence_router is not None:
        app.include_router(intelligence_router)

    # Debug endpoint — shows which routes loaded and which failed (import errors)
    @app.get("/_debug/imports")
    async def debug_imports():
        return {
            "loaded": [k for k in ["forecast", "health", "pnl", "simulation", "auth", "insights", "realtime", "intelligence"] if k not in _IMPORT_ERRORS],
            "failed": _IMPORT_ERRORS,
        }

    return app


app = create_app()
