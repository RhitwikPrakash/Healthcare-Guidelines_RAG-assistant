from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.services.document_store import store

try:
    from rank_bm25 import BM25Okapi as _RankBM25
except ImportError:  # Local verification fallback; Docker installs rank-bm25.
    _RankBM25 = None


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class _FallbackBM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_frequencies = [Counter(document) for document in corpus]
        document_frequency: Counter[str] = Counter()
        for document in corpus:
            document_frequency.update(set(document))
        total_documents = max(1, len(corpus))
        self.idf = {
            term: math.log(1.0 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        for index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            normaliser = self.k1 * (1.0 - self.b + self.b * doc_length / max(self.avgdl, 1.0))
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1.0)) / (frequency + normaliser)
            scores[index] = score
        return scores


BM25Okapi = _RankBM25 or _FallbackBM25Okapi


class BM25Store:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, list[dict[str, Any]], Any]] = {}
        self._lock = threading.RLock()

    def _get(self, session_id: str) -> tuple[list[dict[str, Any]], Any] | None:
        path: Path = store.chunks_path(session_id)
        if not path.exists():
            return None
        modified = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(session_id)
            if cached and cached[0] == modified:
                return cached[1], cached[2]
            chunks = store.load_chunks(session_id)
            if not chunks:
                return None
            corpus = [tokenize(chunk["text"]) for chunk in chunks]
            bm25 = BM25Okapi(corpus, k1=1.45, b=0.72)
            self._cache[session_id] = (modified, chunks, bm25)
            return chunks, bm25

    def invalidate(self, session_id: str) -> None:
        with self._lock:
            self._cache.pop(session_id, None)

    def query(self, session_id: str, query: str, n_results: int) -> list[dict[str, Any]]:
        data = self._get(session_id)
        if not data:
            return []
        chunks, bm25 = data
        scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][: min(n_results, len(chunks))]
        maximum = float(scores[top_indices[0]]) if len(top_indices) else 0.0
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(top_indices, start=1):
            score = float(scores[index])
            if score <= 0 and rank > 3:
                continue
            item = dict(chunks[int(index)])
            item["lexical_score"] = score / maximum if maximum > 0 else 0.0
            item["lexical_rank"] = rank
            results.append(item)
        return results


bm25_store = BM25Store()
