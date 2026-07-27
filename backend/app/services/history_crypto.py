from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


class HistoryCrypto:
    def __init__(self) -> None:
        self._fernet: Fernet | None = None

    def _get_fernet(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet

        settings = get_settings()
        configured = settings.chat_encryption_key.strip().encode("utf-8")
        path: Path = settings.chat_encryption_key_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if configured:
            key = configured
        elif path.exists():
            key = path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            path.write_bytes(key)
            try:
                path.chmod(0o600)
            except OSError:
                pass

        try:
            self._fernet = Fernet(key)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("CHAT_ENCRYPTION_KEY is not a valid Fernet key") from exc
        return self._fernet

    def encrypt_text(self, value: str) -> str:
        return self._get_fernet().encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt_text(self, value: str) -> str:
        # Plaintext fallback supports a safe transition from any earlier local DB.
        try:
            return self._get_fernet().decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeEncodeError, UnicodeDecodeError):
            return value

    def encrypt_json(self, value: dict[str, Any]) -> str:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return self.encrypt_text(raw)

    def decrypt_json(self, value: str) -> dict[str, Any]:
        raw = self.decrypt_text(value)
        try:
            decoded = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}


history_crypto = HistoryCrypto()