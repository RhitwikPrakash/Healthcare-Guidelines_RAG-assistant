from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    label: int
    source: str
    page: int
    text: str


# Small, retrieval-only aliases. They help match common user wording to wording
# used in medical tables without adding medical facts to the generated answer.
_RETRIEVAL_ALIASES: Dict[str, Sequence[str]] = {
    "colorectal": ("colorectal", "colon", "rectum", "bowel"),
    "oral": ("oral", "oral cavity", "mouth"),
    "bladder": ("bladder", "urinary bladder", "urine", "urination"),
    "childhood eye": ("childhood eye", "retinoblastoma", "pupil", "strabismus"),
    "eye cancer": ("eye cancer", "retinoblastoma", "pupil", "strabismus"),
}

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "each",
    "for",
    "from",
    "given",
    "in",
    "is",
    "it",
    "list",
    "main",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "which",
    "with",
}

_GENERIC_ITEM_WORDS = {
    "cancer",
    "cancers",
    "category",
    "categories",
    "type",
    "types",
    "item",
    "items",
}


def split_text(text: str, max_chars: int = 1200, overlap: int = 180) -> List[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + max_chars)
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start = max(start + 1, end - overlap)

    if current:
        chunks.append(current)

    return chunks


def build_chunks(pages: List[Dict]) -> List[Chunk]:
    chunks: List[Chunk] = []
    label = 1

    for page in pages:
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            continue

        for text in split_text(page_text):
            chunks.append(
                Chunk(
                    label=label,
                    source=str(page.get("source") or "document.pdf"),
                    page=int(page.get("page") or 1),
                    text=text,
                )
            )
            label += 1

    return chunks


def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, object]:
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _meaningful_tokens(text: str) -> List[str]:
    return [
        token
        for token in _tokens(text)
        if token not in _QUERY_STOPWORDS and token not in _GENERIC_ITEM_WORDS
    ]


def _extract_requested_items(question: str) -> List[str]:
    """
    Extract an explicit comma/'and' list from a multi-part question.

    Examples:
      "List symptoms for: breast, bladder and prostate cancers"
      "Compare Step 1, Step 2 and Step 3"
    """
    cleaned = " ".join(question.strip().split()).rstrip(".?!")
    candidate = ""

    if ":" in cleaned:
        tail = cleaned.rsplit(":", 1)[1].strip()
        if "," in tail or re.search(r"\band\b", tail, flags=re.IGNORECASE):
            candidate = tail

    if not candidate:
        match = re.search(
            r"\b(?:for|among|across|compare)\b\s+(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match:
            tail = match.group(1).strip()
            if "," in tail or re.search(r"\band\b", tail, flags=re.IGNORECASE):
                candidate = tail

    if not candidate:
        return []

    raw_items = re.split(r"\s*,\s*|\s+\band\b\s+", candidate, flags=re.IGNORECASE)
    items: List[str] = []

    for raw in raw_items:
        item = re.sub(r"^[\s:;-]+|[\s:;.-]+$", "", raw).strip()
        words = [
            word
            for word in _tokens(item)
            if word not in _QUERY_STOPWORDS and word not in _GENERIC_ITEM_WORDS
        ]
        if not words:
            continue

        # Keep a short, readable label for the prompt checklist.
        normalised = " ".join(words[-5:])
        if normalised and normalised not in items:
            items.append(normalised)

    return items if len(items) >= 2 else []


def _item_search_phrases(item: str) -> List[str]:
    phrases = [item]
    lowered = item.casefold()

    for key, aliases in _RETRIEVAL_ALIASES.items():
        if key in lowered or lowered in key:
            phrases.extend(aliases)

    return list(dict.fromkeys(phrase.casefold() for phrase in phrases if phrase.strip()))


def _item_match_score(item: str, chunk_text: str) -> float:
    text = chunk_text.casefold()
    best = 0.0

    for phrase in _item_search_phrases(item):
        if phrase in text:
            best = max(best, 1.0)

        phrase_tokens = _meaningful_tokens(phrase)
        if phrase_tokens:
            token_hits = sum(1 for token in phrase_tokens if token in text)
            best = max(best, token_hits / len(phrase_tokens))

    return best


def _lexical_coverage(question: str, chunk_text: str) -> float:
    query_tokens = set(_meaningful_tokens(question))
    if not query_tokens:
        return 0.0

    chunk_tokens = set(_tokens(chunk_text))
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def retrieve_chunks(
    question: str,
    chunks: List[Chunk],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 8,
) -> List[Chunk]:
    """
    Retrieve semantically relevant chunks while preserving multi-part coverage.

    The previous implementation ranked only the whole question. For questions that
    list several categories, the strongest early categories could dominate the top
    results. This version first reserves the best matching evidence for each explicit
    requested item, then fills the remaining slots by combined semantic/lexical rank.
    """
    if not chunks:
        return []

    safe_top_k = max(1, min(int(top_k), len(chunks)))
    query_vector = vectorizer.transform([question])
    cosine_scores = cosine_similarity(query_vector, matrix).flatten()

    requested_items = _extract_requested_items(question)
    combined_scores = np.array(cosine_scores, dtype=float)

    for index, chunk in enumerate(chunks):
        lexical = _lexical_coverage(question, chunk.text)
        item_coverage = 0.0
        if requested_items:
            matches = [_item_match_score(item, chunk.text) for item in requested_items]
            item_coverage = sum(score >= 0.75 for score in matches) / len(requested_items)

        combined_scores[index] += 0.20 * lexical + 0.35 * item_coverage

    selected_indices: List[int] = []

    # Coverage-first selection for explicit multi-part lists.
    for item in requested_items:
        item_scores = np.array(
            [_item_match_score(item, chunk.text) for chunk in chunks],
            dtype=float,
        )
        best_index = int(np.argmax(item_scores))
        if item_scores[best_index] >= 0.75 and best_index not in selected_indices:
            selected_indices.append(best_index)
        if len(selected_indices) >= safe_top_k:
            break

    ranked_indices = np.argsort(combined_scores)[::-1]
    for index in ranked_indices:
        index = int(index)
        if combined_scores[index] <= 0 and selected_indices:
            continue
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) >= safe_top_k:
            break

    if not selected_indices:
        selected_indices = list(range(safe_top_k))

    # Put the most useful multi-item/table chunk first in the model context.
    selected_indices = sorted(
        selected_indices[:safe_top_k],
        key=lambda index: combined_scores[index],
        reverse=True,
    )

    relabelled: List[Chunk] = []
    for display_label, index in enumerate(selected_indices, start=1):
        chunk = chunks[index]
        relabelled.append(
            Chunk(
                label=display_label,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
            )
        )

    return relabelled


def _preserve_structure(text: str) -> str:
    """
    Compact whitespace without flattening tables/lists into one long sentence.

    Preserving line boundaries is important because rows such as "Urinary bladder"
    and "Retinoblastoma" may otherwise be harder for the model to map correctly.
    """
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines) if lines else " ".join(text.split())


def build_evidence_text(chunks: List[Chunk], max_chars_per_chunk: int = 1400) -> str:
    """
    Build model context while preserving complete 1200-character retrieval chunks.

    split_text() creates chunks up to 1200 characters. The old 850-character cap
    silently removed the end of table/list chunks, which is where later categories
    such as bladder, prostate, and retinoblastoma could appear.
    """
    evidence_parts = []

    for chunk in chunks:
        structured_text = _preserve_structure(chunk.text)
        if len(structured_text) > max_chars_per_chunk:
            structured_text = structured_text[:max_chars_per_chunk].rstrip() + "..."

        evidence_parts.append(
            f"Evidence item {chunk.label}\n"
            f"Source: {chunk.source}\n"
            f"Page: {chunk.page}\n"
            f"Text:\n{structured_text}"
        )

    return "\n\n---\n\n".join(evidence_parts)


def _answer_style(answer_mode: str) -> str:
    if answer_mode == "Fast":
        return (
            "Give a direct answer in one to three short paragraphs. "
            "Prioritize only the most relevant facts."
        )
    if answer_mode == "Deep Research":
        return (
            "Give a structured, comprehensive answer. Compare relevant evidence, "
            "explain important qualifications, and use clear headings when helpful."
        )
    return (
        "Choose an appropriate depth for the question. Be clear, complete, and concise, "
        "using short headings or bullets only when they improve readability."
    )


def build_answer_prompt(question: str, evidence_text: str, answer_mode: str) -> str:
    style = _answer_style(answer_mode)
    requested_items = _extract_requested_items(question)

    checklist = ""
    if requested_items:
        checklist = (
            "\nRequired coverage checklist:\n"
            + "\n".join(f"- {item}" for item in requested_items)
            + "\nYour final answer must contain a separate result or explicit evidence "
            "limitation for every checklist item."
        )

    return f"""
You are answering a question about uploaded healthcare documents.

Grounding rules:
- Answer ONLY from the evidence supplied below.
- Do not use outside knowledge and do not invent facts.
- Keep medical wording educational; do not diagnose or prescribe.
- Do not include citation numbers, bracketed labels, source names, page numbers, a References section, or an Evidence section inside the answer.
- The application will append references and evidence separately after the answer.
- Finish every sentence and do not end mid-thought.

Completeness rules for multi-part questions:
- Silently identify every person, cancer type, step, category, comparison item, or requested sub-question before writing the answer.
- Answer EVERY requested item separately and in the same order as the question.
- Inspect every supplied evidence item before declaring any requested item missing.
- For table evidence, treat each row label and its values as one exact mapping.
- When the question requests a fixed number but the evidence provides fewer, give all available facts and write: "The document lists only X symptoms for this category."
- Use "The uploaded document does not provide enough evidence to answer this." only when no supplied evidence item contains information for that requested item.
- Never replace partial available evidence with a blanket "not enough evidence" statement.
- Before finalising, silently verify that every requested item has a corresponding answer or a precise evidence limitation.
{checklist}

Answer style:
{style}

Question:
{question}

Evidence:
{evidence_text}

Write only the answer body, without citations or references.
""".strip()


def looks_incomplete(answer: str) -> bool:
    if not answer or len(answer.strip()) < 40:
        return True

    text = answer.strip()
    lowered = text.casefold()
    bad_endings = (
        "including",
        "such as",
        "for example",
        "and",
        "or",
        "with",
        "because",
        "therefore",
        "including:",
        "such as:",
    )
    return lowered.endswith(bad_endings) or text[-1] not in ".!?)]}"


def strip_inline_citations(answer: str) -> str:
    """Remove model-produced citation labels and duplicate reference sections."""
    clean = answer.strip()

    clean = re.split(
        r"\n\s*(?:#{1,6}\s*)?(?:references?|sources?|evidence)\s*:\s*\n?",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    clean = re.sub(
        r"\s*\[(?:S?\d+)(?:\s*[,;]\s*S?\d+)*\]",
        "",
        clean,
        flags=re.IGNORECASE,
    )

    clean = re.sub(
        r"\s*\((?:source|evidence)\s*\d+(?:\s*[,;]\s*\d+)*\)",
        "",
        clean,
        flags=re.IGNORECASE,
    )

    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def format_evidence_cards(chunks: List[Chunk]) -> List[Dict]:
    cards: List[Dict] = []

    for chunk in chunks:
        preview = chunk.text.strip()
        if len(preview) > 1200:
            preview = preview[:1200].rstrip() + "..."
        cards.append(
            {
                "label": chunk.label,
                "source": chunk.source,
                "page": chunk.page,
                "text": preview,
            }
        )

    return cards


def build_reference_lines(evidence_cards: List[Dict]) -> List[str]:
    """Create a compact unique source/page list for display after the answer."""
    pages_by_source: Dict[str, List[int]] = {}

    for card in evidence_cards:
        source = str(card.get("source") or "document.pdf")
        page = int(card.get("page") or 1)
        pages_by_source.setdefault(source, [])
        if page not in pages_by_source[source]:
            pages_by_source[source].append(page)

    lines = []
    for source, pages in pages_by_source.items():
        page_text = ", ".join(str(page) for page in sorted(pages))
        lines.append(f"{source} — page(s) {page_text}")

    return lines


# Backward-compatible helpers retained for older local imports/tests.
def valid_citation_labels(answer: str, allowed_labels: List[int]) -> List[int]:
    found = re.findall(r"\[(?:S)?(\d+)\]", answer, flags=re.IGNORECASE)
    allowed = set(allowed_labels)
    return list(dict.fromkeys(int(item) for item in found if int(item) in allowed))


def build_repair_prompt(question: str, draft_answer: str, evidence_text: str) -> str:
    return build_answer_prompt(question, evidence_text, "Smart Auto") + (
        "\n\nRewrite this incomplete draft into a complete answer body without citations:\n"
        + draft_answer
    )


def force_basic_citations(answer: str, evidence_labels: List[int]) -> str:
    del evidence_labels
    return strip_inline_citations(answer)