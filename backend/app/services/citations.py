from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(r"\[S(\d+)\]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
    "may", "can", "should", "could", "would", "also", "than", "when", "which", "who",
}


def build_evidence_context(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    blocks: list[str] = []
    used = 0
    for chunk in chunks:
        source_id = len(selected) + 1
        header = (
            f"[S{source_id}] Source: {chunk['source']} | Page: {chunk['page']} | "
            f"Section: {chunk.get('section', 'Document text')}"
        )
        block = f"{header}\n{chunk['text'].strip()}"
        if selected and used + len(block) > max_chars:
            continue
        selected_item = dict(chunk)
        selected_item["source_id"] = f"S{source_id}"
        selected.append(selected_item)
        blocks.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(blocks), selected


def _claim_blocks(answer: str) -> list[str]:
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        text = " ".join(paragraph).strip()
        if len(_WORD_RE.findall(text)) >= 6:
            blocks.append(text)
        paragraph = []

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            continue
        if line.startswith("> **Urgent safety note:**"):
            flush()
            continue
        if re.match(r"^(?:[-*+] |\d+[.)] )", line):
            flush()
            clean = re.sub(r"^(?:[-*+] |\d+[.)] )", "", line).strip()
            if len(_WORD_RE.findall(clean)) >= 6:
                blocks.append(clean)
            continue
        paragraph.append(line)
    flush()
    return blocks


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(_CITATION_RE.sub("", text))
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _support_overlap(claim: str, evidence: list[dict[str, Any]]) -> float:
    cited = [int(value) - 1 for value in _CITATION_RE.findall(claim)]
    cited_text = " ".join(
        str(evidence[index].get("text", ""))
        for index in cited
        if 0 <= index < len(evidence)
    )
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 1.0
    evidence_tokens = _content_tokens(cited_text)
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def validate_citations(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {str(index) for index in range(1, len(evidence) + 1)}
    found = _CITATION_RE.findall(answer)
    invalid = sorted({item for item in found if item not in valid_ids})
    cleaned = answer
    for item in invalid:
        cleaned = cleaned.replace(f"[S{item}]", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    valid_found = [item for item in _CITATION_RE.findall(cleaned) if item in valid_ids]

    claim_blocks = _claim_blocks(cleaned)
    cited_blocks = [block for block in claim_blocks if _CITATION_RE.search(block)]
    coverage = len(cited_blocks) / len(claim_blocks) if claim_blocks else (1.0 if valid_found else 0.0)
    weakly_supported = [
        block for block in cited_blocks
        if _support_overlap(block, evidence) < 0.06
    ]
    support_ratio = 1.0 - len(weakly_supported) / max(1, len(cited_blocks))

    return {
        "answer": cleaned,
        "valid_citations": sorted({f"S{item}" for item in valid_found}, key=lambda value: int(value[1:])),
        "invalid_citations": [f"S{item}" for item in invalid],
        "coverage": round(coverage, 3),
        "support_ratio": round(support_ratio, 3),
        "claim_blocks": len(claim_blocks),
        "uncited_blocks": len(claim_blocks) - len(cited_blocks),
        "weakly_supported_blocks": len(weakly_supported),
        "needs_repair": not valid_found or coverage < 0.72 or support_ratio < 0.70,
    }


def citation_cards(evidence: list[dict[str, Any]], used_ids: list[str]) -> list[dict[str, Any]]:
    used = set(used_ids)
    cards: list[dict[str, Any]] = []
    for item in evidence:
        if item["source_id"] not in used:
            continue
        cards.append(
            {
                "id": item["source_id"],
                "source": item["source"],
                "page": item["page"],
                "section": item.get("section", "Document text"),
                "excerpt": item["text"][:520].strip(),
                "score": round(float(item.get("final_score", item.get("fusion_score", 0.0))), 4),
            }
        )
    return cards
