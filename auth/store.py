"""
JSON-backed user store for the auth layer.

Users are persisted at ``auth/users.json`` (auto-created and seeded with two
demo accounts on first use). Passwords are never stored in plaintext — only the
PBKDF2 hash and its salt are kept.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from auth.models import Role, UserInDB, UserPublic
from auth.permissions import permissions_for_role
from auth.security import hash_password, verify_password

logger = logging.getLogger("auth")

_STORE_PATH = Path(__file__).resolve().parent / "users.json"

# Demo seed accounts (plaintext passwords used ONLY to derive hashes at seed time).
_DEMO_USERS = [
    {
        "username": "admin",
        "password": "admin123",
        "role": Role.ADMIN,
        "full_name": "Alex Morgan",
        "email": "admin@gic-intelligence.io",
    },
    {
        "username": "user",
        "password": "user123",
        "role": Role.USER,
        "full_name": "Jordan Lee",
        "email": "analyst@gic-intelligence.io",
    },
]


def _role_value(role) -> str:
    return role.value if isinstance(role, Role) else str(role)


class UserStore:
    """Thread-safe, JSON-backed user store."""

    def __init__(self, path: Path = _STORE_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._users: Dict[str, UserInDB] = {}
        self._load_or_seed()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load_or_seed(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self._users = self._read()
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read user store (%s); reseeding.", exc)
            self._seed()

    def _read(self) -> Dict[str, UserInDB]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        users: Dict[str, UserInDB] = {}
        for record in raw.get("users", []):
            user = UserInDB(
                username=record["username"],
                full_name=record["full_name"],
                role=Role(record["role"]),
                email=record["email"],
                password_hash=record["password_hash"],
                salt=record["salt"],
            )
            users[user.username] = user
        return users

    def _write(self) -> None:
        payload = {
            "users": [
                {
                    "username": u.username,
                    "full_name": u.full_name,
                    "role": _role_value(u.role),
                    "email": u.email,
                    "password_hash": u.password_hash,
                    "salt": u.salt,
                }
                for u in self._users.values()
            ]
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _seed(self) -> None:
        self._users = {}
        for seed in _DEMO_USERS:
            pw_hash, salt = hash_password(seed["password"])
            user = UserInDB(
                username=seed["username"],
                full_name=seed["full_name"],
                role=seed["role"],
                email=seed["email"],
                password_hash=pw_hash,
                salt=salt,
            )
            self._users[user.username] = user
        self._write()
        logger.info("Seeded auth user store with %d demo accounts.", len(self._users))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_user(self, username: str) -> Optional[UserInDB]:
        with self._lock:
            return self._users.get(username)

    def verify_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Return the user if credentials are valid, else None."""
        user = self.get_user(username)
        if user is None:
            return None
        if verify_password(password, user.salt, user.password_hash):
            return user
        return None

    def list_users(self) -> List[UserInDB]:
        with self._lock:
            return list(self._users.values())

    def add_user(
        self,
        username: str,
        password: str,
        role: Role,
        full_name: str,
        email: str,
    ) -> UserInDB:
        with self._lock:
            if username in self._users:
                raise ValueError(f"User {username!r} already exists")
            pw_hash, salt = hash_password(password)
            user = UserInDB(
                username=username,
                full_name=full_name,
                role=role if isinstance(role, Role) else Role(role),
                email=email,
                password_hash=pw_hash,
                salt=salt,
            )
            self._users[username] = user
            self._write()
            return user

    def to_public(self, user: UserInDB) -> UserPublic:
        """Project an internal user to its public representation with permissions."""
        return UserPublic(
            username=user.username,
            full_name=user.full_name,
            role=user.role if isinstance(user.role, Role) else Role(user.role),
            email=user.email,
            permissions=permissions_for_role(user.role),
        )


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #
_store_singleton: Optional[UserStore] = None
_singleton_lock = threading.Lock()


def get_user_store() -> UserStore:
    global _store_singleton
    if _store_singleton is None:
        with _singleton_lock:
            if _store_singleton is None:
                _store_singleton = UserStore()
    return _store_singleton


__all__ = ["UserStore", "get_user_store"]
