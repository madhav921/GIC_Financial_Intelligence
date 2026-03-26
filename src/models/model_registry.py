"""
Model registry: save, load, and version trained models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from loguru import logger

from src.config import get_project_root, get_settings


class ModelRegistry:
    """Persist and retrieve trained models with metadata."""

    def __init__(self):
        settings = get_settings()
        self.registry_dir = get_project_root() / settings["paths"]["model_registry"]
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> Path:
        """Save a model artifact with metadata."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / f"{model_name}_{timestamp}.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        meta = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "params": params or {},
            "features": features or [],
            "model_file": model_path.name,
        }
        meta_path = model_dir / f"{model_name}_{timestamp}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update "latest" symlink-like pointer
        latest_path = model_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump({"model_file": model_path.name, "meta_file": meta_path.name}, f, indent=2)

        logger.info(f"Saved model '{model_name}' → {model_path}")
        return model_path

    def load_model(self, model_name: str, version: str | None = None) -> tuple[Any, dict]:
        """Load a model and its metadata. Loads latest if no version specified."""
        model_dir = self.registry_dir / model_name

        if version:
            model_path = model_dir / f"{model_name}_{version}.joblib"
            meta_path = model_dir / f"{model_name}_{version}_meta.json"
        else:
            latest_path = model_dir / "latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"No saved model found for '{model_name}'")
            with open(latest_path) as f:
                latest = json.load(f)
            model_path = model_dir / latest["model_file"]
            meta_path = model_dir / latest["meta_file"]

        model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = json.load(f)

        logger.info(f"Loaded model '{model_name}' from {model_path}")
        return model, meta

    def list_models(self) -> list[dict]:
        """List all registered models with their latest metadata."""
        models = []
        for model_dir in self.registry_dir.iterdir():
            if model_dir.is_dir():
                latest_path = model_dir / "latest.json"
                if latest_path.exists():
                    with open(latest_path) as f:
                        latest = json.load(f)
                    meta_path = model_dir / latest["meta_file"]
                    if meta_path.exists():
                        with open(meta_path) as f:
                            models.append(json.load(f))
        return models
