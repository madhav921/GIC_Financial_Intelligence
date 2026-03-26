"""
FastAPI Application — GIC Plan-to-Perform Engine

Serves the AI-powered financial intelligence platform via REST API.
Integrates all layers of the architecture.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import forecast, health, simulation
from src.config import get_settings
from src.logging_setup import setup_logging


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

    return app


app = create_app()
