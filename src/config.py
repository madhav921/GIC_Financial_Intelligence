"""Configuration loader for the Plan-to-Perform engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"


def _load_env() -> None:
    env_file = _ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load and return the YAML settings dictionary."""
    _load_env()
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton-style cached settings
_settings: dict[str, Any] | None = None


def get_settings() -> dict[str, Any]:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def get_project_root() -> Path:
    return _ROOT


def get_env(key: str, default: str = "") -> str:
    _load_env()
    return os.getenv(key, default)
