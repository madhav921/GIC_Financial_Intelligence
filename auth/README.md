# Auth Layer — Independent RBAC

A **fully decoupled** authentication & authorization layer that sits *over* the
GIC Financial Intelligence platform. It does not import or modify any existing
business logic — the app runs with or without it. Crypto is **stdlib-only**
(PBKDF2-HMAC-SHA256 password hashing, HS256-style HMAC-signed JSON tokens), so
there is nothing extra to install.

## Roles

- **ADMIN** — full access (real simulations, raw data, governance, user mgmt).
- **USER** — read-mostly analyst/viewer plus sandbox simulations.

## Permission matrix

| Permission | ADMIN | USER |
|---|:---:|:---:|
| view_landing | ✅ | ✅ |
| view_dashboard | ✅ | ✅ |
| view_executive_summary | ✅ | ✅ |
| view_forecasts | ✅ | ✅ |
| view_insights | ✅ | ✅ |
| view_aggregated_data | ✅ | ✅ |
| view_market_monitor | ✅ | ✅ |
| view_audit_summary | ✅ | ✅ |
| view_warranty | ✅ | ✅ |
| run_sandbox_simulation | ✅ | ✅ |
| run_simulation | ✅ | — |
| edit_scenarios | ✅ | — |
| view_raw_data | ✅ | — |
| manage_thresholds | ✅ | — |
| trigger_retraining | ✅ | — |
| trigger_data_fetch | ✅ | — |
| view_audit_full | ✅ | — |
| export_reports | ✅ | — |
| regenerate_narratives | ✅ | — |
| manage_users | ✅ | — |

## Demo credentials

| Role | Username | Password |
|---|---|---|
| ADMIN | `admin` | `admin123` |
| USER | `user` | `user123` |

## How it plugs in

In `src/api/app.py`:

```python
from src.api.routes.auth import auth_router
app.include_router(auth_router)
```

Protect any endpoint:

```python
from fastapi import Depends
from auth import require_permission, Permission

@router.post("/simulate")
def simulate(user=Depends(require_permission(Permission.RUN_SIMULATION))):
    ...
```

Endpoints: `POST /auth/login`, `GET /auth/me`, `POST /auth/logout`,
`GET /auth/permissions`, `GET /auth/users` (admin), `GET /auth/demo-profiles`
(public).

This layer is fully decoupled: it lives in its own top-level `auth/` package
plus one router file, and touches no existing application code.
