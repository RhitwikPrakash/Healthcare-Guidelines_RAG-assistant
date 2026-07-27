from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from app.config import get_settings


def safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", session_id).strip("._")
    if len(cleaned) < 4:
        raise ValueError("Invalid session ID")
    return cleaned[:120]


class DocumentStore:
    def __init__(self) -> None:
        self.settings = get_settings()

    def session_dir(self, session_id: str) -> Path:
        return self.settings.sessions_dir / safe_session_id(session_id)

    def upload_dir(self, session_id: str) -> Path:
        """Legacy upload path retained for compatibility and cleanup."""
        return self.settings.uploads_dir / safe_session_id(session_id)

    def active_index_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "active_index.json"

    def active_version(self, session_id: str) -> str | None:
        path = self.active_index_path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = payload.get("version")
            return str(version) if version else None
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def version_metadata_path(self, session_id: str, version: str) -> Path:
        return self.session_dir(session_id) / f"documents.{version}.json"

    def version_chunks_path(self, session_id: str, version: str) -> Path:
        return self.session_dir(session_id) / f"chunks.{version}.jsonl"

    def version_upload_dir(self, session_id: str, version: str) -> Path:
        return self.settings.uploads_dir / safe_session_id(session_id) / version

    def metadata_path(self, session_id: str) -> Path:
        version = self.active_version(session_id)
        if version:
            return self.version_metadata_path(session_id, version)
        return self.session_dir(session_id) / "documents.json"

    def chunks_path(self, session_id: str) -> Path:
        version = self.active_version(session_id)
        if version:
            return self.version_chunks_path(session_id, version)
        return self.session_dir(session_id) / "chunks.jsonl"

    def write_uploaded_file(self, session_id: str, file_name: str, data: bytes) -> Path:
        safe_name = Path(file_name).name
        if not safe_name.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files are supported: {safe_name}")
        target_dir = self.upload_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / safe_name
        target.write_bytes(data)
        return target

    def write_index_payload(
        self,
        session_id: str,
        documents: list[dict[str, Any]],
        chunks: Iterable[dict[str, Any]],
    ) -> None:
        """Write a legacy payload, mainly used by lightweight tests."""
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "documents.json").write_text(
            json.dumps(documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (session_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def write_version_payload(
        self,
        session_id: str,
        version: str,
        documents: list[dict[str, Any]],
        chunks: Iterable[dict[str, Any]],
    ) -> None:
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        metadata_target = self.version_metadata_path(session_id, version)
        chunks_target = self.version_chunks_path(session_id, version)
        metadata_tmp = metadata_target.with_suffix(metadata_target.suffix + ".tmp")
        chunks_tmp = chunks_target.with_suffix(chunks_target.suffix + ".tmp")
        metadata_tmp.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        metadata_tmp.replace(metadata_target)
        chunks_tmp.replace(chunks_target)

    def list_documents(self, session_id: str) -> list[dict[str, Any]]:
        path = self.metadata_path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def load_chunks(self, session_id: str) -> list[dict[str, Any]]:
        path = self.chunks_path(session_id)
        if not path.exists():
            return []
        chunks: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        chunks.append(item)
        except OSError:
            return []
        return chunks

    def remove_version(self, session_id: str, version: str) -> None:
        for path in (
            self.version_metadata_path(session_id, version),
            self.version_chunks_path(session_id, version),
        ):
            try:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(self.version_upload_dir(session_id, version), ignore_errors=True)

    def cleanup_versions(self, session_id: str, keep_version: str) -> None:
        session_dir = self.session_dir(session_id)
        if session_dir.exists():
            for path in session_dir.glob("documents.*.json"):
                version = path.name[len("documents.") : -len(".json")]
                if version != keep_version:
                    self.remove_version(session_id, version)
            for path in session_dir.glob("chunks.*.jsonl"):
                version = path.name[len("chunks.") : -len(".jsonl")]
                if version != keep_version:
                    self.remove_version(session_id, version)
        upload_root = self.settings.uploads_dir / safe_session_id(session_id)
        if upload_root.exists():
            for child in upload_root.iterdir():
                if child.is_dir() and child.name != keep_version:
                    shutil.rmtree(child, ignore_errors=True)

    def clear(self, session_id: str) -> None:
        shutil.rmtree(self.session_dir(session_id), ignore_errors=True)
        shutil.rmtree(self.upload_dir(session_id), ignore_errors=True)

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


store = DocumentStore()
