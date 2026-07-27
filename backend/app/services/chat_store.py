from __future__ import annotations
import sqlite3
from typing import Any
from uuid import uuid4
from app.config import get_settings
from app.database import db
from app.services.history_crypto import history_crypto

_DEFAULT_TITLE = "New research chat"

def _conversation_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": history_crypto.decrypt_text(str(row["title"])),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "message_count": int(row["message_count"] if "message_count" in row.keys() else 0),
    }


def _message_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "role": str(row["role"]),
        "content": history_crypto.decrypt_text(str(row["content"])),
        "metadata": history_crypto.decrypt_json(str(row["metadata_json"] or "{}")),
        "created_at": str(row["created_at"]),
    }

class ChatStore:
    def create_conversation(self, user_id: str, title: str = _DEFAULT_TITLE) -> dict[str, Any]:
        conversation_id = str(uuid4())
        clean_title = " ".join(title.strip().split())[:120] or _DEFAULT_TITLE
        db.execute(
            "INSERT INTO conversations(id, user_id, title) VALUES (?, ?, ?)",
            (conversation_id, user_id, history_crypto.encrypt_text(clean_title)),
        )
        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            raise RuntimeError("The conversation could not be created")
        return conversation

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        rows = db.fetch_all(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (user_id,),
        )
        return [_conversation_from_row(row) for row in rows]

    def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        row = db.fetch_one(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.id = ? AND c.user_id = ?
            GROUP BY c.id
            """,
            (conversation_id, user_id),
        )
        return _conversation_from_row(row) if row else None

    def is_owner(self, user_id: str, conversation_id: str) -> bool:
        return self.get_conversation(user_id, conversation_id) is not None

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> dict[str, Any] | None:
        clean_title = " ".join(title.strip().split())[:120]
        if not clean_title:
            raise ValueError("Conversation title cannot be empty")
        changed = db.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (history_crypto.encrypt_text(clean_title), conversation_id, user_id),
        )
        return self.get_conversation(user_id, conversation_id) if changed else None

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        return bool(
            db.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        )

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported message role")
        if not self.is_owner(user_id, conversation_id):
            raise PermissionError("Conversation not found")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Message content cannot be empty")

        final_message_id = message_id or str(uuid4())
        encrypted_content = history_crypto.encrypt_text(clean_content)
        encrypted_metadata = history_crypto.encrypt_json(metadata or {})
        with db.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO messages(id, conversation_id, role, content, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (final_message_id, conversation_id, role, encrypted_content, encrypted_metadata),
                )
                if role == "user":
                    row = connection.execute(
                        "SELECT title FROM conversations WHERE id = ? AND user_id = ?",
                        (conversation_id, user_id),
                    ).fetchone()
                    current_title = history_crypto.decrypt_text(str(row["title"])) if row else ""
                    if current_title == _DEFAULT_TITLE:
                        automatic_title = " ".join(clean_content.split())[:80]
                        connection.execute(
                            "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
                            (history_crypto.encrypt_text(automatic_title), conversation_id, user_id),
                        )
                connection.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                )
                row = connection.execute(
                    "SELECT id, role, content, metadata_json, created_at FROM messages WHERE id = ?",
                    (final_message_id,),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if not row:
            raise RuntimeError("The message could not be saved")
        return _message_from_row(row)

    def get_message(self, user_id: str, conversation_id: str, message_id: str) -> dict[str, Any] | None:
        row = db.fetch_one(
            """
            SELECT m.id, m.role, m.content, m.metadata_json, m.created_at
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.id = ? AND m.conversation_id = ? AND c.user_id = ?
            """,
            (message_id, conversation_id, user_id),
        )
        return _message_from_row(row) if row else None

    def message_exists(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        *,
        role: str | None = None,
    ) -> bool:
        message = self.get_message(user_id, conversation_id, message_id)
        return bool(message and (role is None or message["role"] == role))

    def list_messages(self, user_id: str, conversation_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        if not self.is_owner(user_id, conversation_id):
            raise PermissionError("Conversation not found")
        safe_limit = max(1, min(int(limit), 5000))
        rows = db.fetch_all(
            """
            SELECT id, role, content, metadata_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, rowid ASC
            LIMIT ?
            """,
            (conversation_id, safe_limit),
        )
        return [_message_from_row(row) for row in rows]

    def previous_user_question(self, user_id: str, conversation_id: str) -> str | None:
        """Return only the latest user question. Assistant output is never reused as model context."""
        if not self.is_owner(user_id, conversation_id):
            raise PermissionError("Conversation not found")
        rows = db.fetch_all(
            """
            SELECT id, role, content, metadata_json, created_at
            FROM messages
            WHERE conversation_id = ? AND role = 'user'
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (conversation_id,),
        )
        if not rows:
            return None
        return str(_message_from_row(rows[0])["content"])

    def recent_assistant_answers(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 6,
    ) -> list[str]:
        """Return recent verified answers for post-generation contamination checks only.

        These strings must never be placed in an Ollama generation prompt.
        """
        if not self.is_owner(user_id, conversation_id):
            raise PermissionError("Conversation not found")
        safe_limit = max(1, min(int(limit), 20))
        rows = db.fetch_all(
            """
            SELECT id, role, content, metadata_json, created_at
            FROM messages
            WHERE conversation_id = ? AND role = 'assistant'
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, safe_limit * 3),
        )
        answers: list[str] = []
        for row in rows:
            message = _message_from_row(row)
            metadata = message.get("metadata") or {}
            if not metadata.get("request_id") or not metadata.get("question_hash"):
                continue
            if not bool((metadata.get("quality") or {}).get("passed", False)):
                continue
            answers.append(str(message["content"]))
            if len(answers) >= safe_limit:
                break
        return answers

    def recent_history(self, user_id: str, conversation_id: str, limit: int = 8) -> list[dict[str, str]]:
        """Compatibility helper returning user text only, never assistant answers."""
        previous = self.previous_user_question(user_id, conversation_id)
        return [{"role": "user", "content": previous}] if previous else []

    def add_bound_assistant_message(
        self,
        user_id: str,
        conversation_id: str,
        content: str,
        metadata: dict[str, Any],
        *,
        parent_message_id: str,
        request_id: str,
        question_hash: str,
    ) -> dict[str, Any]:
        """Atomically save one assistant response beside its exact originating question."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Message content cannot be empty")
        if not all((parent_message_id, request_id, question_hash)):
            raise ValueError("A complete answer binding is required")

        final_metadata = dict(metadata)
        final_metadata.update(
            {
                "request_id": request_id,
                "question_hash": question_hash,
                "in_reply_to_message_id": parent_message_id,
            }
        )
        encrypted_content = history_crypto.encrypt_text(clean_content)
        encrypted_metadata = history_crypto.encrypt_json(final_metadata)
        assistant_id = str(uuid4())

        with db.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                owner_row = connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()
                if not owner_row:
                    raise PermissionError("Conversation not found")

                parent_row = connection.execute(
                    """
                    SELECT id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE id = ? AND conversation_id = ? AND role = 'user'
                    """,
                    (parent_message_id, conversation_id),
                ).fetchone()
                if not parent_row:
                    raise RuntimeError("The originating question no longer exists")
                parent = _message_from_row(parent_row)
                parent_metadata = parent.get("metadata") or {}
                if str(parent_metadata.get("request_id") or "") != request_id:
                    raise RuntimeError("The originating question request binding changed")
                if str(parent_metadata.get("question_hash") or "") != question_hash:
                    raise RuntimeError("The originating question hash changed")

                existing_rows = connection.execute(
                    """
                    SELECT id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE conversation_id = ? AND role = 'assistant'
                    ORDER BY created_at DESC, rowid DESC
                    """,
                    (conversation_id,),
                ).fetchall()
                for row in existing_rows:
                    existing = _message_from_row(row)
                    existing_metadata = existing.get("metadata") or {}
                    same_request = str(existing_metadata.get("request_id") or "") == request_id
                    same_parent = str(existing_metadata.get("in_reply_to_message_id") or "") == parent_message_id
                    if same_request or same_parent:
                        existing_hash = str(existing_metadata.get("question_hash") or "")
                        if existing_hash != question_hash:
                            raise RuntimeError("A conflicting assistant response already exists")
                        connection.rollback()
                        return existing

                connection.execute(
                    """
                    INSERT INTO messages(id, conversation_id, role, content, metadata_json)
                    VALUES (?, ?, 'assistant', ?, ?)
                    """,
                    (assistant_id, conversation_id, encrypted_content, encrypted_metadata),
                )
                connection.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                )
                row = connection.execute(
                    "SELECT id, role, content, metadata_json, created_at FROM messages WHERE id = ?",
                    (assistant_id,),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        if not row:
            raise RuntimeError("The assistant response could not be saved")
        return _message_from_row(row)

    def clear_messages(self, user_id: str, conversation_id: str) -> bool:
        if not self.is_owner(user_id, conversation_id):
            return False
        with db.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                    """,
                    (history_crypto.encrypt_text(_DEFAULT_TITLE), conversation_id, user_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return True

    def cleanup_expired_history(self) -> dict[str, int]:
        months = get_settings().chat_retention_months
        modifier = f"-{months} months"
        with db.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                message_cursor = connection.execute(
                    "DELETE FROM messages WHERE datetime(created_at) < datetime('now', ?)",
                    (modifier,),
                )
                conversation_cursor = connection.execute(
                    """
                    DELETE FROM conversations
                    WHERE datetime(updated_at) < datetime('now', ?)
                      AND NOT EXISTS (
                          SELECT 1 FROM messages WHERE messages.conversation_id = conversations.id
                      )
                    """,
                    (modifier,),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "messages_deleted": max(0, int(message_cursor.rowcount)),
            "conversations_deleted": max(0, int(conversation_cursor.rowcount)),
        }


chat_store = ChatStore()