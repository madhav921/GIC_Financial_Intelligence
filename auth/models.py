"""
Data models for the auth layer: roles, permissions, and request/response
schemas.

Pydantic v2 is the platform's validation library and is used here for the
schemas. If, for any reason, pydantic cannot be imported, the schemas fall back
to lightweight dataclasses with a compatible constructor surface so the layer
still loads.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional


class Role(str, Enum):
    """User roles. Extend by adding members and updating ROLE_PERMISSIONS."""

    ADMIN = "admin"
    USER = "user"


class Permission(str, Enum):
    """Granular permissions. UPPER_SNAKE names, lower_snake string values."""

    VIEW_LANDING = "view_landing"
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_EXECUTIVE_SUMMARY = "view_executive_summary"
    VIEW_FORECASTS = "view_forecasts"
    VIEW_INSIGHTS = "view_insights"
    VIEW_AGGREGATED_DATA = "view_aggregated_data"
    VIEW_MARKET_MONITOR = "view_market_monitor"
    VIEW_AUDIT_SUMMARY = "view_audit_summary"
    VIEW_WARRANTY = "view_warranty"
    RUN_SANDBOX_SIMULATION = "run_sandbox_simulation"
    RUN_SIMULATION = "run_simulation"
    EDIT_SCENARIOS = "edit_scenarios"
    VIEW_RAW_DATA = "view_raw_data"
    MANAGE_THRESHOLDS = "manage_thresholds"
    TRIGGER_RETRAINING = "trigger_retraining"
    TRIGGER_DATA_FETCH = "trigger_data_fetch"
    VIEW_AUDIT_FULL = "view_audit_full"
    EXPORT_REPORTS = "export_reports"
    REGENERATE_NARRATIVES = "regenerate_narratives"
    MANAGE_USERS = "manage_users"


# --------------------------------------------------------------------------- #
# Schemas — pydantic v2 preferred, dataclass fallback.
# --------------------------------------------------------------------------- #
try:
    from pydantic import BaseModel, Field

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - exercised only if pydantic is absent
    _HAS_PYDANTIC = False


if _HAS_PYDANTIC:

    class LoginRequest(BaseModel):
        username: str
        password: str

    class UserPublic(BaseModel):
        username: str
        full_name: str
        role: Role
        email: str
        permissions: List[str] = Field(default_factory=list)

    class UserInDB(BaseModel):
        username: str
        full_name: str
        role: Role
        email: str
        password_hash: str
        salt: str

    class TokenResponse(BaseModel):
        access_token: str
        token_type: str = "bearer"
        expires_in: int
        user: UserPublic

else:  # pragma: no cover - fallback path
    from dataclasses import dataclass, field

    @dataclass
    class LoginRequest:
        username: str
        password: str

    @dataclass
    class UserPublic:
        username: str
        full_name: str
        role: Role
        email: str
        permissions: List[str] = field(default_factory=list)

    @dataclass
    class UserInDB:
        username: str
        full_name: str
        role: Role
        email: str
        password_hash: str
        salt: str

    @dataclass
    class TokenResponse:
        access_token: str
        expires_in: int
        user: "UserPublic"
        token_type: str = "bearer"


__all__ = [
    "Role",
    "Permission",
    "LoginRequest",
    "UserPublic",
    "UserInDB",
    "TokenResponse",
]
