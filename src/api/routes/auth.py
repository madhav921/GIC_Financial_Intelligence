"""
Authentication & authorization API routes.

This router is the single integration point of the otherwise self-contained
``auth/`` layer. Wire it in from ``src/api/app.py`` with::

    from src.api.routes.auth import auth_router
    app.include_router(auth_router)

Heavy imports are wrapped so the module still loads (with the router disabled)
if a dependency is somehow missing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

try:
    from auth.config import get_auth_settings
    from auth.models import LoginRequest, TokenResponse, UserPublic
    from auth.permissions import permissions_for_role
    from auth.security import create_access_token
    from auth.store import get_user_store
    from auth.dependencies import get_current_user, require_admin

    _AUTH_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - defensive: keep module importable
    _AUTH_AVAILABLE = False
    _IMPORT_ERROR = exc


auth_router = APIRouter(prefix="/auth", tags=["auth"])


# Demo profile metadata for frontend quick-login buttons.
_DEMO_PROFILES = [
    {
        "label": "Administrator",
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "description": "Full access — simulations, raw data, governance",
    },
    {
        "label": "Analyst (Viewer)",
        "username": "user",
        "password": "user123",
        "role": "user",
        "description": "Read-only dashboards & insights",
    },
]


if _AUTH_AVAILABLE:

    @auth_router.post("/login", response_model=TokenResponse)
    def login(payload: LoginRequest) -> TokenResponse:
        """Verify credentials and issue a signed access token."""
        store = get_user_store()
        user = store.verify_user(payload.username, payload.password)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        settings = get_auth_settings()
        role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
        token = create_access_token({"sub": user.username, "role": role_value})
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=settings.token_expiry_minutes * 60,
            user=store.to_public(user),
        )

    @auth_router.get("/me", response_model=UserPublic)
    def me(user: UserPublic = Depends(get_current_user)) -> UserPublic:
        """Return the currently authenticated user's public profile."""
        return user

    @auth_router.post("/logout")
    def logout() -> dict:
        """Stateless logout. The client simply drops the token."""
        return {"detail": "logged out"}

    @auth_router.get("/permissions")
    def my_permissions(user: UserPublic = Depends(get_current_user)) -> dict:
        """Return the caller's role and resolved permission list."""
        role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
        return {"role": role_value, "permissions": permissions_for_role(user.role)}

    @auth_router.get("/users", response_model=list[UserPublic])
    def list_users(_: UserPublic = Depends(require_admin)) -> list[UserPublic]:
        """List all users (admin only)."""
        store = get_user_store()
        return [store.to_public(u) for u in store.list_users()]

    @auth_router.get("/demo-profiles")
    def demo_profiles() -> list[dict]:
        """Public: demo profiles for frontend quick-login buttons.

        Passwords are included only when DEMO_MODE is enabled.
        """
        settings = get_auth_settings()
        result = []
        for profile in _DEMO_PROFILES:
            entry = {
                "label": profile["label"],
                "username": profile["username"],
                "role": profile["role"],
                "description": profile["description"],
            }
            if settings.demo_mode:
                entry["demo_password"] = profile["password"]
            result.append(entry)
        return result

else:  # pragma: no cover - only hit if auth imports fail

    @auth_router.get("/_unavailable")
    def _unavailable() -> dict:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth layer unavailable: {_IMPORT_ERROR}",
        )


__all__ = ["auth_router"]
