"""
FastAPI dependencies for the auth layer.

These wire the stdlib-only token + user store machinery into FastAPI's
dependency-injection system. They are imported lazily by ``auth/__init__.py`` so
that the rest of the package can be used without FastAPI installed.
"""

from __future__ import annotations

from typing import Callable, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.models import Permission, Role, UserPublic
from auth.permissions import has_permission
from auth.security import decode_access_token
from auth.store import get_user_store

# Bearer-token extractor. ``auto_error=False`` lets us return clean 401s and
# support optional-auth endpoints.
_bearer = HTTPBearer(auto_error=False, description="Bearer access token")
oauth2_scheme = _bearer

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def _resolve_user(creds: Optional[HTTPAuthorizationCredentials]) -> UserPublic:
    if creds is None or not creds.credentials:
        raise _UNAUTHENTICATED
    try:
        payload = decode_access_token(creds.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    username = payload.get("sub")
    if not username:
        raise _UNAUTHENTICATED

    store = get_user_store()
    user = store.get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return store.to_public(user)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserPublic:
    """Resolve the authenticated user from the bearer token, or 401."""
    return _resolve_user(creds)


def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[UserPublic]:
    """Like :func:`get_current_user` but returns ``None`` instead of raising.

    Useful for endpoints whose response varies by role but are still reachable
    anonymously.
    """
    if creds is None or not creds.credentials:
        return None
    try:
        return _resolve_user(creds)
    except HTTPException:
        return None


def require_permission(permission: Permission) -> Callable[..., UserPublic]:
    """Build a dependency that 403s unless the user's role grants ``permission``."""

    def _dependency(user: UserPublic = Depends(get_current_user)) -> UserPublic:
        if not has_permission(user.role, permission):
            perm_value = (
                permission.value if isinstance(permission, Permission) else str(permission)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {perm_value}",
            )
        return user

    return _dependency


def require_admin(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    """Shortcut dependency: require the ADMIN role."""
    role = user.role if isinstance(user.role, Role) else Role(user.role)
    if role is not Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required",
        )
    return user


__all__ = [
    "oauth2_scheme",
    "get_current_user",
    "get_optional_user",
    "require_permission",
    "require_admin",
]
