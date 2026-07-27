from __future__ import annotations

import re
from collections import Counter
from typing import Any


_SPACE_RE = re.compile(r"[ \t]+")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "").replace("\u00a0", " ")
    return _SPACE_RE.sub(" ", line).strip()


def canonical_margin_line(line: str) -> str:
    line = normalize_line(line).lower()
    line = re.sub(r"\d+", "#", line)
    return line[:180]


def remove_repeated_margins(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(pages) < 3:
        return pages

    margin_candidates: list[str] = []
    per_page_lines: list[list[str]] = []
    for page in pages:
        lines = [normalize_line(line) for line in page["text"].splitlines() if normalize_line(line)]
        per_page_lines.append(lines)
        margin_candidates.extend(canonical_margin_line(line) for line in (lines[:2] + lines[-2:]))

    counts = Counter(margin_candidates)
    threshold = max(3, int(len(pages) * 0.45))
    repeated = {line for line, count in counts.items() if count >= threshold and len(line) >= 4}

    cleaned: list[dict[str, Any]] = []
    for page, lines in zip(pages, per_page_lines):
        kept: list[str] = []
        for index, line in enumerate(lines):
            at_margin = index < 2 or index >= max(0, len(lines) - 2)
            if at_margin and canonical_margin_line(line) in repeated:
                continue
            if _PAGE_NUMBER_RE.match(line):
                continue
            kept.append(line)
        text = "\n".join(kept)
        cleaned.append({**page, "text": clean_text(text)})
    return cleaned


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [normalize_line(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
