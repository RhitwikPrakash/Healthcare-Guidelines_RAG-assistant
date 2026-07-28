from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthUser:
    username: str
    email: str


def _read_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return str(value).strip()

    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value is not None:
                return str(value).strip()
    except Exception:
        pass

    return None


def auth_is_configured() -> bool:
    return bool(
        _read_secret("AUTH_USERNAME")
        and _read_secret("AUTH_EMAIL")
        and (_read_secret("AUTH_PASSWORD") or _read_secret("AUTH_PASSWORD_HASH"))
    )


def hash_password(password: str, *, iterations: int = 260_000) -> str:
    """Return a PBKDF2-SHA256 password hash suitable for Streamlit Secrets."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password_hash(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (TypeError, ValueError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def authenticate(username: str, email: str, password: str) -> Optional[AuthUser]:
    configured_username = _read_secret("AUTH_USERNAME")
    configured_email = _read_secret("AUTH_EMAIL")
    configured_password = _read_secret("AUTH_PASSWORD")
    configured_password_hash = _read_secret("AUTH_PASSWORD_HASH")

    if not configured_username or not configured_email:
        return None

    username_matches = hmac.compare_digest(
        username.strip().casefold(),
        configured_username.casefold(),
    )
    email_matches = hmac.compare_digest(
        email.strip().casefold(),
        configured_email.casefold(),
    )

    if configured_password_hash:
        password_matches = _verify_password_hash(password, configured_password_hash)
    elif configured_password is not None:
        password_matches = hmac.compare_digest(password, configured_password)
    else:
        password_matches = False

    if not (username_matches and email_matches and password_matches):
        return None

    return AuthUser(username=configured_username, email=configured_email)