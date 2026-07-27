from uuid import uuid4

from app.services.document_store import store
from app.services.hybrid_retriever import retriever


def test_section_coverage_spans_documents_and_late_sections():
    session_id = f"coverage-{uuid4().hex}"
    chunks = []
    try:
        for source in ("a.pdf", "b.pdf"):
            for index in range(10):
                chunks.append(
                    {
                        "id": f"{source}-{index}",
                        "source": source,
                        "page": index + 1,
                        "section": f"Section {index}",
                        "chunk_index": index,
                        "text": f"Evidence for {source} section {index}",
                    }
                )
        store.write_index_payload(session_id, [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}], chunks)
        samples = retriever.section_coverage(session_id, maximum=6)
        assert len(samples) == 6
        assert {item["source"] for item in samples} == {"a.pdf", "b.pdf"}
        assert max(int(item["page"]) for item in samples) >= 9
    finally:
        store.clear(session_id)
