"""
Vercel Python ASGI entry point.

Vercel looks for an ASGI/WSGI callable in api/index.py.  This module
re-exports the FastAPI application instance so Vercel can discover it
without any further configuration.

Usage:
    This file is imported automatically by @vercel/python.
    Do NOT invoke directly — use `uvicorn src.api.app:app` for local dev.
"""

from src.api.app import app  # noqa: F401 — Vercel ASGI entry point

__all__ = ["app"]
