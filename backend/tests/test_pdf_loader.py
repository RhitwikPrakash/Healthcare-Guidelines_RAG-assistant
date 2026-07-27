from pathlib import Path

import fitz

from app.services.pdf_loader import load_pdf


def test_pdf_text_extraction(tmp_path: Path):
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Clinical guideline recommendation for adults with hypertension and follow-up care.")
    document.save(path)
    document.close()

    pages = load_pdf(path)
    assert len(pages) == 1
    assert "hypertension" in pages[0]["text"].lower()
