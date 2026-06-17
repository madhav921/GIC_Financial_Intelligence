"""
Configuration for the independent auth layer.

All settings are resolved from environment variables with safe development
defaults. Nothing here imports application code, so the layer stays decoupled.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("auth")

# Development fallback secret. NEVER rely on this in production — set
# AUTH_SECRET_KEY in the environment instead.
_DEV_SECRET_KEY = "dev-insecure-auth-secret-change-me-in-production"

ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRY_MINUTES = 480


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid int for %s=%r; using default %d", name, raw, default)
        return default


@dataclass(frozen=True)
class AuthSettings:
    secret_key: str
    token_expiry_minutes: int
    algorithm: str
    demo_mode: bool


def get_auth_settings() -> AuthSettings:
    """Build the auth settings from the environment (re-read each call)."""
    secret = os.getenv("AUTH_SECRET_KEY")
    if not secret:
        logger.warning(
            "AUTH_SECRET_KEY is not set; using an insecure development secret. "
            "Set AUTH_SECRET_KEY in the environment for production use."
        )
        secret = _DEV_SECRET_KEY

    return AuthSettings(
        secret_key=secret,
        token_expiry_minutes=_env_int("AUTH_TOKEN_EXPIRY_MINUTES", DEFAULT_TOKEN_EXPIRY_MINUTES),
        algorithm=ALGORITHM,
        demo_mode=_env_bool("AUTH_DEMO_MODE", True),
    )
