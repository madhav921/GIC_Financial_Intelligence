"""
Role -> permission matrix and helpers.

ADMIN receives every permission. USER receives a read-mostly subset plus the
sandbox simulation capability. Extend by editing ``_USER_PERMISSIONS`` or adding
new roles to ``ROLE_PERMISSIONS``.
"""

from __future__ import annotations

from typing import Dict, List, Set, Union

from auth.models import Permission, Role

# Permissions granted to the USER (analyst / viewer) role.
_USER_PERMISSIONS: Set[Permission] = {
    Permission.VIEW_LANDING,
    Permission.VIEW_DASHBOARD,
    Permission.VIEW_EXECUTIVE_SUMMARY,
    Permission.VIEW_FORECASTS,
    Permission.VIEW_INSIGHTS,
    Permission.VIEW_AGGREGATED_DATA,
    Permission.VIEW_MARKET_MONITOR,
    Permission.VIEW_AUDIT_SUMMARY,
    Permission.VIEW_WARRANTY,
    Permission.RUN_SANDBOX_SIMULATION,
}

# ADMIN gets everything.
_ADMIN_PERMISSIONS: Set[Permission] = set(Permission)

ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: _ADMIN_PERMISSIONS,
    Role.USER: _USER_PERMISSIONS,
}


def _coerce_role(role: Union[Role, str]) -> Role:
    if isinstance(role, Role):
        return role
    return Role(role)


def _coerce_permission(permission: Union[Permission, str]) -> Permission:
    if isinstance(permission, Permission):
        return permission
    return Permission(permission)


def has_permission(role: Union[Role, str], permission: Union[Permission, str]) -> bool:
    """Return True if ``role`` is granted ``permission``."""
    try:
        r = _coerce_role(role)
        p = _coerce_permission(permission)
    except ValueError:
        return False
    return p in ROLE_PERMISSIONS.get(r, set())


def permissions_for_role(role: Union[Role, str]) -> List[str]:
    """Return the sorted list of permission string values for ``role``."""
    try:
        r = _coerce_role(role)
    except ValueError:
        return []
    return sorted(p.value for p in ROLE_PERMISSIONS.get(r, set()))


__all__ = ["ROLE_PERMISSIONS", "has_permission", "permissions_for_role"]
