from __future__ import annotations

import json
import re
from typing import Any

from app.config import get_settings
from app.services.ollama_client import ollama


_WHOLE_DOCUMENT_RE = re.compile(
    r"\b(?:whole|entire|all sections|across the document|complete summary|summari[sz]e the document|overview of the (?:pdf|guideline|paper))\b",
    re.IGNORECASE,
)
_COMPLEX_RE = re.compile(
    r"\b(?:compare|contrast|relationship|causes? and|diagnosis and|treatment and|benefits? and risks?|all recommendations|eligibility|contraindications|monitoring)\b",
    re.IGNORECASE,
)


class QueryPlanner:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _heuristic(self, question: str, mode: str) -> dict[str, Any]:
        whole_document = bool(_WHOLE_DOCUMENT_RE.search(question))
        complex_question = (
            mode == "deep"
            or whole_document
            or bool(_COMPLEX_RE.search(question))
            or len(question.split()) >= 24
            or question.count("?") > 1
        )
        subqueries = [question.strip()]
        if complex_question:
            clauses = re.split(r"\b(?:and|versus|vs\.?|compared with|while)\b", question, flags=re.IGNORECASE)
            for clause in clauses:
                clause = clause.strip(" ,.;:?")
                if len(clause.split()) >= 4 and clause.lower() != question.lower():
                    subqueries.append(clause)
        return {
            "complex": complex_question,
            "whole_document": whole_document,
            "subqueries": list(dict.fromkeys(subqueries))[:4],
            "strategy": "document-wide" if whole_document else ("multi-hop" if complex_question else "focused"),
            "planner": "deterministic",
        }

    def plan(self, question: str, mode: str) -> dict[str, Any]:
        base = self._heuristic(question, mode)
        should_use_llm = (
            self.settings.planner_mode == "always"
            or (self.settings.planner_mode == "auto" and base["complex"] and mode != "fast")
        )
        if not should_use_llm or self.settings.testing:
            return base

        prompt = (
            "Create a retrieval plan for a healthcare-guideline PDF question. "
            "Return JSON only with keys: subqueries (1-5 short standalone search queries), "
            "whole_document (boolean), strategy (focused|multi-hop|document-wide). "
            "Do not answer the medical question. Preserve drug names, populations, outcomes, and constraints.\n\n"
            f"Question: {question}"
        )
        try:
            response = ollama.chat_with_fallback(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                preferred_model=self.settings.llm_fallback_model,
                max_tokens=400,
            )
            payload = json.loads(response["content"])
            subqueries = [str(item).strip() for item in payload.get("subqueries", []) if str(item).strip()]
            if not subqueries:
                return base
            return {
                "complex": len(subqueries) > 1 or base["complex"],
                "whole_document": bool(payload.get("whole_document", base["whole_document"])),
                "subqueries": list(dict.fromkeys(subqueries))[:5],
                "strategy": str(payload.get("strategy") or base["strategy"]),
                "planner": "ollama",
                "planner_model": response["model"],
                "planner_seconds": response["elapsed_seconds"],
            }
        except Exception as exc:  # noqa: BLE001
            base["planner_warning"] = str(exc)
            return base


planner = QueryPlanner()
