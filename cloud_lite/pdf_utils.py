import re
from typing import List, Dict

import fitz


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(file_bytes: bytes, filename: str) -> List[Dict]:
    pages = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text("text")
            text = clean_text(text)

            if text:
                pages.append(
                    {
                        "source": filename,
                        "page": page_index + 1,
                        "text": text,
                    }
                )

    return pages