"""Health check and system info routes."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.config import get_settings
from src.models.model_registry import ModelRegistry

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    try:
        registry = ModelRegistry()
        n_models = len(registry.list_models())
    except Exception:
        n_models = 0

    return HealthResponse(
        status="ok",
        version=settings["project"]["version"],
        models_loaded=n_models,
    )


@router.get("/models")
async def list_models():
    """List all registered models."""
    registry = ModelRegistry()
    return {"models": registry.list_models()}
