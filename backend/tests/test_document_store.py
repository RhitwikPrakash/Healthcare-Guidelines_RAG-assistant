from uuid import uuid4

from app.services.document_store import store


def test_versioned_payload_activates_atomically():
    session_id = f"test-{uuid4().hex}"
    try:
        old_version = "old"
        new_version = "new"
        store.write_version_payload(
            session_id,
            old_version,
            [{"file_name": "old.pdf", "pages": 1, "chunks": 1}],
            [{"id": "old-1", "text": "old evidence"}],
        )
        store.active_index_path(session_id).write_text('{"version":"old"}', encoding="utf-8")
        assert store.list_documents(session_id)[0]["file_name"] == "old.pdf"
        assert store.load_chunks(session_id)[0]["id"] == "old-1"

        store.write_version_payload(
            session_id,
            new_version,
            [{"file_name": "new.pdf", "pages": 2, "chunks": 1}],
            [{"id": "new-1", "text": "new evidence"}],
        )
        # A staged version must not become visible before pointer activation.
        assert store.list_documents(session_id)[0]["file_name"] == "old.pdf"

        pointer = store.active_index_path(session_id)
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text('{"version":"new"}', encoding="utf-8")
        temporary.replace(pointer)
        assert store.list_documents(session_id)[0]["file_name"] == "new.pdf"
        assert store.load_chunks(session_id)[0]["id"] == "new-1"
    finally:
        store.clear(session_id)
