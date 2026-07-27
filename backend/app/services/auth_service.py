from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.database import db


_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_password_hasher = PasswordHasher(time_cost=3, memory_cost=32768, parallelism=2)
_bearer = HTTPBearer(auto_error=False)


def _normalise_email(email: str) -> str:
    value = email.strip().lower()
    if not _EMAIL_PATTERN.fullmatch(value):
        raise ValueError("Enter a valid email address")
    return value


def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "email": str(row["email"]),
        "display_name": str(row["display_name"]),
        "created_at": str(row["created_at"]),
    }


class AuthService:
    def __init__(self) -> None:
        self._secret: str | None = None

    def _jwt_secret(self) -> str:
        if self._secret:
            return self._secret
        settings = get_settings()
        configured = settings.jwt_secret.strip()
        if configured:
            if len(configured) < 32:
                raise RuntimeError("JWT_SECRET must contain at least 32 characters")
            self._secret = configured
            return configured

        path: Path = settings.jwt_secret_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            secret = path.read_text(encoding="utf-8").strip()
        else:
            secret = secrets.token_urlsafe(48)
            path.write_text(secret, encoding="utf-8")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        if len(secret) < 32:
            raise RuntimeError("The generated JWT secret is invalid")
        self._secret = secret
        return secret

    def register(self, email: str, display_name: str, password: str) -> dict[str, Any]:
        normalised_email = _normalise_email(email)
        clean_name = " ".join(display_name.strip().split())
        if len(clean_name) < 2:
            raise ValueError("Display name must contain at least two characters")
        if len(password) < 8:
            raise ValueError("Password must contain at least eight characters")

        user_id = str(uuid4())
        password_hash = _password_hasher.hash(password)
        try:
            db.execute(
                "INSERT INTO users(id, email, display_name, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, normalised_email, clean_name, password_hash),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with this email already exists") from exc
        user = self.get_user(user_id)
        if not user:
            raise RuntimeError("The user account could not be created")
        return user

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        try:
            normalised_email = _normalise_email(email)
        except ValueError:
            return None
        row = db.fetch_one(
            "SELECT id, email, display_name, password_hash, created_at, is_active FROM users WHERE email = ?",
            (normalised_email,),
        )
        if not row or not bool(row["is_active"]):
            return None
        try:
            _password_hasher.verify(str(row["password_hash"]), password)
        except (VerifyMismatchError, InvalidHashError):
            return None
        if _password_hasher.check_needs_rehash(str(row["password_hash"])):
            db.execute(
                "UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (_password_hasher.hash(password), str(row["id"])),
            )
        return _row_to_user(row)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        row = db.fetch_one(
            "SELECT id, email, display_name, created_at FROM users WHERE id = ? AND is_active = 1",
            (user_id,),
        )
        return _row_to_user(row) if row else None

    def create_access_token(self, user: dict[str, Any]) -> tuple[str, int]:
        settings = get_settings()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=settings.jwt_expire_days)
        payload = {
            "sub": user["id"],
            "email": user["email"],
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "iss": settings.jwt_issuer,
        }
        token = jwt.encode(payload, self._jwt_secret(), algorithm="HS256")
        return token, int((expires - now).total_seconds())

    def decode_access_token(self, token: str) -> dict[str, Any]:
        settings = get_settings()
        try:
            return jwt.decode(
                token,
                self._jwt_secret(),
                algorithms=["HS256"],
                issuer=settings.jwt_issuer,
                options={"require": ["sub", "iat", "exp", "iss"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Your login session is invalid or has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc


auth = AuthService()


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = auth.decode_access_token(credentials.credentials)
    user = auth.get_user(str(payload["sub"]))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This account is unavailable",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user