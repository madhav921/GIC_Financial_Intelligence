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

    llm_status = None
    try:
        from layers.layer5_governance.controller import GovernanceLayerController
        llm_status = GovernanceLayerController().llm_health_check()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=settings["project"]["version"],
        models_loaded=n_models,
        llm_status=llm_status,
    )


@router.get("/models")
async def list_models():
    """List all registered models."""
    registry = ModelRegistry()
    return {"models": registry.list_models()}
