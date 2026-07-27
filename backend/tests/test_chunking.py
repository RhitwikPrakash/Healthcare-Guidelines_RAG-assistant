from app.services.chunking import chunk_pages, is_heading


def test_heading_detection():
    assert is_heading("3.2 Treatment recommendations")
    assert is_heading("RECOMMENDATIONS")
    assert not is_heading("Treatment was associated with lower blood pressure.")


def test_section_aware_chunks_keep_metadata():
    pages = [
        {
            "source": "guide.pdf",
            "page": 4,
            "text": "RECOMMENDATIONS\nAdults with hypertension should receive structured follow-up. " * 10,
        }
    ]
    chunks = chunk_pages(pages)
    assert chunks
    assert all(chunk["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk["page"] == 4 for chunk in chunks)
    assert any("RECOMMENDATIONS" in chunk["section"] for chunk in chunks)


def test_section_heading_carries_across_pages():
    pages = [
        {"source": "guide.pdf", "page": 1, "text": "TREATMENT RECOMMENDATIONS\nStart therapy after confirmation."},
        {"source": "guide.pdf", "page": 2, "text": "Continue monitoring blood pressure and adverse effects."},
    ]
    chunks = chunk_pages(pages)
    second_page = [chunk for chunk in chunks if chunk["page"] == 2]
    assert second_page
    assert all(chunk["section"] == "TREATMENT RECOMMENDATIONS" for chunk in second_page)
