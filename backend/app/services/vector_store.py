from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.services.document_store import store
from app.services.embeddings import embeddings


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._lock = threading.RLock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                try:
                    import chromadb
                except ImportError as exc:
                    raise RuntimeError(
                        "ChromaDB is not installed. Start the project with Docker or install backend requirements."
                    ) from exc
                self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            return self._client

    @staticmethod
    def _prefix(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def collection_name(cls, session_id: str, version: str) -> str:
        suffix = hashlib.sha256(version.encode("utf-8")).hexdigest()[:12]
        return f"healthrag_{cls._prefix(session_id)}_{suffix}"

    @staticmethod
    def _active_path(session_id: str) -> Path:
        return store.active_index_path(session_id)

    def active_version(self, session_id: str) -> str | None:
        return store.active_version(session_id)

    def index_staged(
        self,
        session_id: str,
        version: str,
        chunks: list[dict[str, Any]],
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        client = self._get_client()
        name = self.collection_name(session_id, version)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "embedding_model": embeddings.name, "version": version},
        )
        batch_size = 24
        total = len(chunks)
        try:
            for start in range(0, total, batch_size):
                batch = chunks[start : start + batch_size]
                vectors = embeddings.encode([chunk["text"] for chunk in batch], batch_size=12)
                metadatas = [
                    {
                        "source": chunk["source"],
                        "page": int(chunk["page"]),
                        "section": chunk["section"],
                        "chunk_index": int(chunk["chunk_index"]),
                    }
                    for chunk in batch
                ]
                collection.add(
                    ids=[chunk["id"] for chunk in batch],
                    embeddings=vectors,
                    documents=[chunk["text"] for chunk in batch],
                    metadatas=metadatas,
                )
                if progress:
                    progress(min(start + len(batch), total), total)
        except Exception:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            raise
        return total

    def discard_version(self, session_id: str, version: str) -> None:
        try:
            self._get_client().delete_collection(self.collection_name(session_id, version))
        except Exception:
            pass

    def activate(self, session_id: str, version: str) -> None:
        path = self._active_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        old_version = self.active_version(session_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
        temporary.replace(path)
        if old_version and old_version != version:
            try:
                self._get_client().delete_collection(self.collection_name(session_id, old_version))
            except Exception:
                pass

    def clear(self, session_id: str) -> None:
        prefix = f"healthrag_{self._prefix(session_id)}_"
        client = self._get_client()
        try:
            collections = client.list_collections()
        except Exception:
            collections = []
        for collection in collections:
            name = collection.name if hasattr(collection, "name") else str(collection)
            if name.startswith(prefix):
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
        try:
            self._active_path(session_id).unlink(missing_ok=True)
        except OSError:
            pass

    def query(self, session_id: str, query: str, n_results: int) -> list[dict[str, Any]]:
        version = self.active_version(session_id)
        if not version:
            return []
        try:
            collection = self._get_client().get_collection(self.collection_name(session_id, version))
        except Exception:
            return []
        count = collection.count()
        if count == 0:
            return []
        vector = embeddings.encode([query])[0]
        payload = collection.query(
            query_embeddings=[vector],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        results: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(ids):
            distance = float(distances[rank]) if rank < len(distances) else 1.0
            results.append(
                {
                    "id": chunk_id,
                    "text": documents[rank],
                    **(metadatas[rank] or {}),
                    "dense_score": max(0.0, 1.0 - distance),
                    "dense_rank": rank + 1,
                }
            )
        return results


vector_store = VectorStore()
