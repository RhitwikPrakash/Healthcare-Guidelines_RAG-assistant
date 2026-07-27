from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import fitz

from app.services.text_cleaner import clean_text, remove_repeated_margins


ProgressCallback = Callable[[int, int, str], None]


def _page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks", sort=True)
    text_blocks: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = clean_text(str(block[4]))
        if text:
            text_blocks.append(text)
    if text_blocks:
        return "\n\n".join(text_blocks)
    return clean_text(page.get_text("text", sort=True))


def load_pdf(path: Path, progress: ProgressCallback | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty PDF: {path.name}")

    pages: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDFs are not supported: {path.name}")
        total = document.page_count
        if total == 0:
            raise ValueError(f"No pages found in {path.name}")
        for index in range(total):
            page = document.load_page(index)
            text = _page_text(page)
            pages.append(
                {
                    "source": path.name,
                    "page": index + 1,
                    "text": text,
                    "has_selectable_text": bool(text.strip()),
                }
            )
            if progress:
                progress(index + 1, total, path.name)

    pages = remove_repeated_margins(pages)
    text_chars = sum(len(page["text"]) for page in pages)
    selectable_pages = sum(1 for page in pages if page["text"].strip())
    if text_chars < 80 or selectable_pages == 0:
        raise ValueError(
            f"Very little selectable text was found in {path.name}. "
            "This PDF may be scanned; OCR is intentionally not run automatically."
        )
    return pages
