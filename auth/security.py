"""
Stdlib-only cryptographic primitives for the auth layer.

- Password hashing: PBKDF2-HMAC-SHA256 (200,000 iterations).
- Access tokens: HS256-style HMAC-SHA256 signed JSON tokens in the compact
  ``base64url(header).base64url(payload).base64url(signature)`` form.

No third-party crypto dependencies (no pyjwt / passlib / bcrypt).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Optional, Tuple

from auth.config import get_auth_settings

_PBKDF2_ITERATIONS = 200_000
_PBKDF2_HASH = "sha256"
_SALT_BYTES = 16


# --------------------------------------------------------------------------- #
# base64url helpers
# --------------------------------------------------------------------------- #
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _b64url_encode_json(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _b64url_encode(raw)


def _b64url_decode_json(segment: str) -> dict:
    return json.loads(_b64url_decode(segment).decode("utf-8"))


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash ``password`` with PBKDF2-HMAC-SHA256.

    Returns ``(hash_hex, salt_hex)``. A new random salt is generated when one is
    not supplied.
    """
    if salt is None:
        salt_bytes = os.urandom(_SALT_BYTES)
    else:
        salt_bytes = bytes.fromhex(salt)

    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_HASH,
        password.encode("utf-8"),
        salt_bytes,
        _PBKDF2_ITERATIONS,
    )
    return dk.hex(), salt_bytes.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Constant-time verification of ``password`` against stored hash + salt."""
    try:
        candidate_hex, _ = hash_password(password, salt=salt_hex)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate_hex, hash_hex)


# --------------------------------------------------------------------------- #
# Access tokens (HS256-style HMAC-signed JSON)
# --------------------------------------------------------------------------- #
def _sign(signing_input: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return _b64url_encode(sig)


def create_access_token(payload: Dict, expires_minutes: Optional[int] = None) -> str:
    """Create a signed access token.

    The token payload always carries ``iat`` and ``exp`` (unix seconds). The
    caller's ``payload`` typically includes ``sub`` (username) and ``role``.
    """
    settings = get_auth_settings()
    if expires_minutes is None:
        expires_minutes = settings.token_expiry_minutes

    now = int(time.time())
    body = dict(payload)
    body.setdefault("iat", now)
    body["exp"] = now + int(expires_minutes) * 60

    header = {"alg": settings.algorithm, "typ": "JWT"}
    header_seg = _b64url_encode_json(header)
    payload_seg = _b64url_encode_json(body)
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii")
    signature_seg = _sign(signing_input, settings.secret_key)
    return f"{header_seg}.{payload_seg}.{signature_seg}"


def decode_access_token(token: str) -> Dict:
    """Verify signature + expiry and return the token payload.

    Raises ``ValueError`` if the token is malformed, the signature is invalid,
    or the token has expired.
    """
    settings = get_auth_settings()
    if not token or not isinstance(token, str):
        raise ValueError("Empty or invalid token")

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed token: expected 3 segments")

    header_seg, payload_seg, signature_seg = parts
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii")
    expected_sig = _sign(signing_input, settings.secret_key)

    if not hmac.compare_digest(expected_sig, signature_seg):
        raise ValueError("Invalid token signature")

    try:
        payload = _b64url_decode_json(payload_seg)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Malformed token payload") from exc

    exp = payload.get("exp")
    if exp is None:
        raise ValueError("Token missing expiry")
    if int(time.time()) >= int(exp):
        raise ValueError("Token has expired")

    return payload


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
]
