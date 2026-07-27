from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.config import get_settings
from app.services.bm25_store import bm25_store
from app.services.document_store import store
from app.services.reranker import reranker
from app.services.vector_store import vector_store


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(self, session_id: str, query: str, deep: bool = False) -> list[dict[str, Any]]:
        dense_k = self.settings.dense_k + (12 if deep else 0)
        lexical_k = self.settings.lexical_k + (12 if deep else 0)
        dense = vector_store.query(session_id, query, dense_k)
        lexical = bm25_store.query(session_id, query, lexical_k)

        merged: dict[str, dict[str, Any]] = {}
        rrf: defaultdict[str, float] = defaultdict(float)
        for item in dense:
            merged[item["id"]] = dict(item)
            rrf[item["id"]] += 1.0 / (60.0 + item["dense_rank"])
        for item in lexical:
            existing = merged.setdefault(item["id"], dict(item))
            existing.update({key: value for key, value in item.items() if key not in existing})
            existing["lexical_score"] = item.get("lexical_score", 0.0)
            existing["lexical_rank"] = item.get("lexical_rank")
            rrf[item["id"]] += 1.0 / (60.0 + item["lexical_rank"])

        if not merged:
            return []
        maximum = max(rrf.values()) or 1.0
        candidates = []
        for chunk_id, item in merged.items():
            item["fusion_score"] = rrf[chunk_id] / maximum
            candidates.append(item)
        candidates.sort(key=lambda item: item["fusion_score"], reverse=True)
        candidates = candidates[: self.settings.rerank_k + (8 if deep else 0)]
        return reranker.rerank(query, candidates)[: self.settings.final_k + (4 if deep else 0)]

    def section_coverage(self, session_id: str, maximum: int = 12) -> list[dict[str, Any]]:
        chunks = store.load_chunks(session_id)
        representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            source = str(chunk["source"])
            section = str(chunk.get("section", "Document text"))
            key = (source, section)
            if key in seen:
                continue
            seen.add(key)
            candidate = dict(chunk)
            candidate["fusion_score"] = 0.35
            candidate["coverage_sample"] = True
            representatives[source].append(candidate)

        # Round-robin and evenly spaced selection prevents early sections or the first
        # uploaded PDF from dominating a document-wide research request.
        sampled_by_source: dict[str, list[dict[str, Any]]] = {}
        source_count = max(1, len(representatives))
        per_source = max(1, maximum // source_count)
        for source, items in representatives.items():
            if len(items) <= per_source:
                sampled_by_source[source] = items
                continue
            if per_source == 1:
                indices = [len(items) // 2]
            else:
                indices = [round(index * (len(items) - 1) / (per_source - 1)) for index in range(per_source)]
            sampled_by_source[source] = [items[index] for index in dict.fromkeys(indices)]

        selected: list[dict[str, Any]] = []
        sources = list(sampled_by_source)
        cursor = 0
        while len(selected) < maximum and sources:
            source = sources[cursor % len(sources)]
            items = sampled_by_source[source]
            if items:
                selected.append(items.pop(0))
            else:
                sources.remove(source)
                cursor -= 1
            cursor += 1
        return selected

    def expand_neighbours(self, session_id: str, selected: list[dict[str, Any]], radius: int = 1) -> list[dict[str, Any]]:
        chunks = store.load_chunks(session_id)
        by_source_index = {(item["source"], int(item["chunk_index"])): item for item in chunks}
        expanded: dict[str, dict[str, Any]] = {item["id"]: dict(item) for item in selected}
        for item in selected[:5]:
            source = item["source"]
            index = int(item.get("chunk_index", 0))
            for offset in range(-radius, radius + 1):
                neighbour = by_source_index.get((source, index + offset))
                if neighbour and neighbour["id"] not in expanded:
                    copy = dict(neighbour)
                    copy["neighbour_of"] = item["id"]
                    copy["final_score"] = max(0.0, item.get("final_score", item.get("fusion_score", 0.0)) - 0.08)
                    expanded[copy["id"]] = copy
        return sorted(
            expanded.values(),
            key=lambda item: item.get("final_score", item.get("fusion_score", 0.0)),
            reverse=True,
        )


retriever = HybridRetriever()
