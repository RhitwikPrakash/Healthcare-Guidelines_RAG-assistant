"""Private, per-Google-user 72-hour chat-history storage for Streamlit.

Only the Streamlit SERVER talks to Supabase. Never put a Supabase secret API
key in client-side code, a public Git repo, a browser, or a chat message.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests


def _secret(name: str) -> str:
    import os
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:
        import streamlit as st
        return str(st.secrets.get(name, "") or "").strip()
    except (KeyError, FileNotFoundError):
        return ""


def history_owner_from_google_claims(claims: dict[str, Any]) -> str | None:
    """Stable pseudonymous user key from a Google verified ID-token subject.

    st.user is populated by Streamlit's verified OIDC flow; never use the
    email/name typed by an end user as a storage authorization credential.
    """
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        return None
    # Hash instead of storing raw Google subject or user's email in DB.
    return hashlib.sha256(("google-oidc:" + subject).encode("utf-8")).hexdigest()


class HistoryStore:
    TABLE = "rag_chat_history"

    def __init__(self, url: str = "", secret_key: str = ""):
        parsed = urlparse(url.rstrip("/"))
        self.url = url.rstrip("/")
        self.key = secret_key.strip()
        # Reject mistaken non-TLS or non-Supabase endpoints to avoid key leaks.
        self.is_configured = bool(
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.hostname.endswith(".supabase.co")
            and parsed.path in ("", "/")
            and self.key.startswith("sb_secret_")
        )

    @classmethod
    def from_config(cls) -> "HistoryStore":
        return cls(_secret("SUPABASE_URL"), _secret("SUPABASE_SECRET_KEY"))

    def _request(self, method: str, *, params=None, payload=None, extra_headers=None):
        if not self.is_configured:
            raise RuntimeError("3-day history database is not configured.")
        # New-format Supabase keys belong in apikey, never in Bearer headers.
        headers = {
            "apikey": self.key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        try:
            result = requests.request(
                method,
                self.url + "/rest/v1/" + self.TABLE,
                headers=headers,
                params=params,
                json=payload,
                timeout=(5, 15),
            )
            result.raise_for_status()
            return result.json() if result.content else []
        except requests.RequestException as exc:
            # Do not display server error bodies, which may contain sensitive data.
            raise RuntimeError("Secure chat-history storage request failed.") from exc

    def load_chats(self, owner_id: str) -> list[dict]:
        # DB removes old rows hourly, but expired rows are also hidden here.
        now = datetime.now(timezone.utc).isoformat()
        rows = self._request(
            "GET",
            params={
                "select": "id,title,messages,created_at,updated_at",
                "owner_id": "eq." + owner_id,
                "expires_at": "gt." + now,
                "order": "updated_at.desc",
                "limit": "100",
            },
        )
        result = []
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            safe_messages = [item for item in messages if isinstance(item, dict)
                             and item.get("role") in ("assistant", "user")
                             and isinstance(item.get("content"), str)]
            result.append({
                "id": str(row["id"]),
                "title": str(row.get("title") or "Chat")[:60],
                "messages": safe_messages,
                "notice": None,
                "created_at": str(row.get("created_at") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            })
        return result

    def upsert_chat(self, owner_id: str, chat: dict) -> None:
        # We store chat text, references and selected evidence snippets only.
        # PDF bytes, TF-IDF index and extracted full document are never stored.
        self._request(
            "POST",
            params={"on_conflict": "id"},
            payload={
                "id": chat["id"],
                "owner_id": owner_id,
                "title": chat.get("title", "Chat")[:60],
                "messages": chat.get("messages", []),
            },
            extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    def delete_chat(self, owner_id: str, chat_id: str) -> None:
        self._request(
            "DELETE",
            params={"id": "eq." + chat_id, "owner_id": "eq." + owner_id},
            extra_headers={"Prefer": "return=minimal"},
        )
