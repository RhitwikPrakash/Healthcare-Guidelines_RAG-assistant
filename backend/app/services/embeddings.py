from __future__ import annotations

import hashlib
import math
import threading
from typing import Iterable

import numpy as np
import requests

from app.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        if self.settings.testing:
            return "deterministic-test-embedder"
        if self.settings.embedding_backend == "ollama":
            return self.settings.ollama_embedding_model
        return self.settings.embedding_model

    def _load_sentence_transformer(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    cache_folder=str(self.settings.hf_home),
                    device="cpu",
                    trust_remote_code=False,
                )
            return self._model

    def encode(self, texts: Iterable[str], batch_size: int = 16) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        if self.settings.testing:
            return [self._deterministic_vector(text) for text in items]
        if self.settings.embedding_backend == "ollama":
            return self._encode_ollama(items)

        model = self._load_sentence_transformer()
        vectors = model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32).tolist()

    def _encode_ollama(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/embed"
        response = requests.post(
            url,
            json={"model": self.settings.ollama_embedding_model, "input": texts},
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an invalid embeddings payload")
        return [self._normalise(vector) for vector in vectors]

    @staticmethod
    def _normalise(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector)) or 1.0
        return [float(value) / norm for value in vector]

    @staticmethod
    def _deterministic_vector(text: str, dimensions: int = 256) -> list[float]:
        vector = np.zeros(dimensions, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).tolist()


embeddings = EmbeddingService()
