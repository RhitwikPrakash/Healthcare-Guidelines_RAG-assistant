from __future__ import annotations
import os
from typing import Any
import requests

class RAGApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
        self.token: str | None = None
        self.session = requests.Session()

    def set_token(self, token: str | None) -> None:
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _check(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or response.reason}
        if not response.ok:
            raise RuntimeError(str(payload.get("detail") or payload))
        return payload

    def _request(self, method: str, path: str, *, auth: bool = True, timeout: int = 20, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if auth:
            headers.update(self._headers())
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
        return self._check(response)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", auth=False, timeout=12)

    def config(self) -> dict[str, Any]:
        return self._request("GET", "/config", auth=False, timeout=12)

    def register(self, email: str, display_name: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/register",
            auth=False,
            json={"email": email, "display_name": display_name, "password": password},
        )

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/login",
            auth=False,
            json={"email": email, "password": password},
        )

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def conversations(self) -> list[dict[str, Any]]:
        return self._request("GET", "/conversations")

    def create_conversation(self, title: str = "New research chat") -> dict[str, Any]:
        return self._request("POST", "/conversations", json={"title": title})

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        return self._request("PATCH", f"/conversations/{conversation_id}", json={"title": title})

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/conversations/{conversation_id}", timeout=60)

    def messages(self, conversation_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        return self._request("GET", f"/conversations/{conversation_id}/messages", params={"limit": limit})

    def clear_messages(self, conversation_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/conversations/{conversation_id}/messages")

    def documents(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{session_id}")

    def clear_documents(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{session_id}", timeout=60)

    def start_upload(self, session_id: str, uploaded_files: list[Any]) -> str:
        files = [("files", (item.name, item.getvalue(), "application/pdf")) for item in uploaded_files]
        payload = self._request(
            "POST",
            "/documents/upload",
            data={"session_id": session_id},
            files=files,
            timeout=120,
        )
        return str(payload["job_id"])

    def start_query(self, session_id: str, question: str, mode: str, request_id: str) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/query",
            json={
                "session_id": session_id,
                "request_id": request_id,
                "question": question,
                "mode": mode,
            },
            timeout=30,
        )
        required = ("job_id", "request_id", "conversation_id", "user_message_id", "question_hash", "document_set_hash")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise RuntimeError(f"Backend returned an incomplete query binding: {', '.join(missing)}")
        return payload

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}")

api = RAGApiClient()