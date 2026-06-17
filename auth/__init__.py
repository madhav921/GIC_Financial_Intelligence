"""
Independent Authentication & Authorization (RBAC) Layer
=======================================================

This package is a *fully decoupled* security layer that sits OVER the GIC
Financial Intelligence platform. It does not modify, import, or depend on any
existing business logic — the rest of the application continues to work whether
or not this layer is wired in.

What it provides
----------------
- Two roles (``ADMIN``, ``USER``) with a granular, extensible permission matrix.
- Stdlib-only cryptography (no third-party crypto deps):
    * Password hashing via :func:`hashlib.pbkdf2_hmac` (SHA-256, 200k iters).
    * HS256-style HMAC-signed JSON access tokens (base64url
      ``header.payload.signature`` using :mod:`hmac` + :mod:`hashlib`).
- A JSON-backed user store seeded with two demo accounts.
- Ready-to-use FastAPI dependencies (``get_current_user``, ``require_permission``,
  ``require_admin``) and a self-contained router.

How to plug it in
-----------------
In ``src/api/app.py`` (wired in separately, not by this layer)::

    from src.api.routes.auth import auth_router
    app.include_router(auth_router)

Protect any endpoint::

    from auth import require_permission, Permission

    @router.post("/simulate")
    def simulate(user=Depends(require_permission(Permission.RUN_SIMULATION))):
        ...

Public exports
--------------
``Role``, ``Permission``, ``has_permission``, ``permissions_for_role``,
``UserStore``, ``get_user_store``, ``create_access_token``,
``decode_access_token``, ``get_current_user``, ``require_permission``,
``require_admin``, ``get_optional_user``, and ``auth_router`` (lazy).
"""

from __future__ import annotations

from auth.models import Role, Permission
from auth.permissions import has_permission, permissions_for_role
from auth.security import create_access_token, decode_access_token
from auth.store import UserStore, get_user_store

__all__ = [
    "Role",
    "Permission",
    "has_permission",
    "permissions_for_role",
    "UserStore",
    "get_user_store",
    "create_access_token",
    "decode_access_token",
    # Lazily resolved (require FastAPI):
    "get_current_user",
    "require_permission",
    "require_admin",
    "get_optional_user",
    "auth_router",
]


def __getattr__(name: str):
    """Lazy attribute access for FastAPI-dependent symbols.

    This avoids importing FastAPI (and the router, which can create a circular
    import) unless these names are actually requested.
    """
    if name in {"get_current_user", "require_permission", "require_admin", "get_optional_user"}:
        from auth import dependencies

        return getattr(dependencies, name)
    if name == "auth_router":
        from src.api.routes.auth import auth_router

        return auth_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
