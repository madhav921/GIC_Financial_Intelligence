"""Logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config import get_project_root, get_settings


def setup_logging() -> None:
    """Configure loguru logger with console sink; add file sink only when writable."""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    # File sink — skipped on read-only filesystems (e.g. Vercel serverless)
    try:
        log_dir = get_project_root() / settings["paths"]["logs"]
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "app.log",
            rotation="10 MB",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        )
    except OSError:
        pass
    logger.info("Logging initialised")
