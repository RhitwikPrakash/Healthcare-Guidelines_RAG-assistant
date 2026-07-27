from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from app.config import get_settings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainSplitter
except ImportError:  # Lightweight verification fallback; Docker installs LangChain.
    _LangChainSplitter = None


_HEADING_PATTERNS = (
    re.compile(r"^\d+(?:\.\d+)*\s+[A-Z][^.!?]{2,120}$"),
    re.compile(r"^[A-Z][A-Z0-9 /&(),:\-]{4,120}$"),
    re.compile(
        r"^(?:abstract|summary|executive summary|background|introduction|methods?|results?|discussion|conclusion|recommendations?|references|appendix|annex)\b.*$",
        re.IGNORECASE,
    ),
)


class _FallbackRecursiveSplitter:
    """Small dependency-free splitter used only when LangChain is unavailable."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str], **_: Any) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size // 2))
        self.separators = separators

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[start : start + self.chunk_size] for start in range(0, len(text), step)]

        separator = separators[0]
        pieces = text.split(separator) if separator else list(text)
        if len(pieces) == 1:
            return self._split_recursive(text, separators[1:])

        groups: list[str] = []
        current = ""
        joiner = separator
        for piece in pieces:
            candidate = piece if not current else f"{current}{joiner}{piece}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                groups.extend(self._split_recursive(current, separators[1:]))
            current = piece
        if current:
            groups.extend(self._split_recursive(current, separators[1:]))
        return groups

    def split_text(self, text: str) -> list[str]:
        base = self._split_recursive(text, self.separators)
        if self.chunk_overlap <= 0 or len(base) <= 1:
            return base
        overlapped: list[str] = []
        previous_tail = ""
        for piece in base:
            merged = f"{previous_tail} {piece}".strip() if previous_tail else piece
            overlapped.append(merged[: self.chunk_size])
            previous_tail = piece[-self.chunk_overlap :]
        return overlapped


def _make_splitter(chunk_size: int, chunk_overlap: int):
    kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": ["\n\n", "\n", ". ", "; ", ", ", " "],
        "length_function": len,
        "is_separator_regex": False,
    }
    if _LangChainSplitter is not None:
        return _LangChainSplitter(**kwargs)
    return _FallbackRecursiveSplitter(**kwargs)


def is_heading(line: str) -> bool:
    candidate = " ".join(line.split()).strip()
    if len(candidate) < 3 or len(candidate) > 140:
        return False
    if candidate.endswith((".", ";", "?")):
        return False
    return any(pattern.match(candidate) for pattern in _HEADING_PATTERNS)


def _section_blocks(text: str, initial_heading: str = "Document text") -> tuple[list[tuple[str, str]], str]:
    # Carry the most recent heading across page boundaries. Clinical guideline sections
    # commonly span several pages, so resetting metadata on each page harms retrieval.
    current_heading = initial_heading
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))
        current_lines = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if is_heading(line):
            flush()
            current_heading = line
        else:
            current_lines.append(line)
    flush()
    return (sections or [(current_heading, text)], current_heading)


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = get_settings()
    splitter = _make_splitter(settings.chunk_size, settings.chunk_overlap)

    chunks: list[dict[str, Any]] = []
    source_counters: defaultdict[str, int] = defaultdict(int)
    source_headings: defaultdict[str, str] = defaultdict(lambda: "Document text")
    for page in pages:
        source = str(page["source"])
        page_number = int(page["page"])
        sections, latest_heading = _section_blocks(str(page["text"]), source_headings[source])
        source_headings[source] = latest_heading
        for section, body in sections:
            pieces = splitter.split_text(body)
            for local_index, piece in enumerate(pieces):
                piece = piece.strip()
                if len(piece) < 40:
                    continue
                global_index = source_counters[source]
                source_counters[source] += 1
                digest = hashlib.sha1(
                    f"{source}|{page_number}|{section}|{global_index}|{piece}".encode("utf-8")
                ).hexdigest()[:20]
                chunks.append(
                    {
                        "id": digest,
                        "text": piece,
                        "source": source,
                        "page": page_number,
                        "section": section,
                        "chunk_index": global_index,
                        "local_index": local_index,
                        "char_count": len(piece),
                    }
                )
    if not chunks:
        raise ValueError("No usable chunks were produced from the uploaded PDFs")
    return chunks
