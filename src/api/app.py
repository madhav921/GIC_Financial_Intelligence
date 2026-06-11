"""
FastAPI Application — GIC Plan-to-Perform Engine

Serves the AI-powered financial intelligence platform via REST API.
Integrates all layers of the architecture.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import forecast, health, pnl, simulation
from src.config import get_settings
from src.logging_setup import setup_logging

# Optional layers — wired defensively so the core API still boots if a
# layer's dependencies are unavailable in a given environment.
try:
    from src.api.routes.auth import auth_router
except Exception:  # pragma: no cover - defensive import
    auth_router = None

try:
    from src.api.routes.insights import insights_router
except Exception:  # pragma: no cover - defensive import
    insights_router = None


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

    # Register routes
    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(simulation.router)
    app.include_router(pnl.router)

    # Auth (RBAC) layer — independent, additive
    if auth_router is not None:
        app.include_router(auth_router)

    # Actionable-intelligence layer — insights, variance bridge, warranty
    if insights_router is not None:
        app.include_router(insights_router)

    return app


app = create_app()
