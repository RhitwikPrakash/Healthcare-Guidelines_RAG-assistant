from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.services.bm25_store import bm25_store
from app.services.chunking import chunk_pages
from app.services.document_store import store
from app.services.job_manager import jobs
from app.services.pdf_loader import load_pdf
from app.services.vector_store import vector_store


UPLOAD_STEPS = [
    "Validate PDF files",
    "Extract page text",
    "Clean and section the documents",
    "Create medical embeddings",
    "Build hybrid indexes",
    "Activate the verified knowledge base",
]


def ingest_job(job_id: str, session_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    settings = get_settings()
    started = time.perf_counter()
    version = str(uuid4())
    staging_root = settings.data_dir / "staging" / job_id
    staging_uploads = staging_root / "uploads"
    staging_uploads.mkdir(parents=True, exist_ok=True)
    activated = False

    try:
        jobs.update(job_id, step=0, phase="Validating PDFs", detail="Checking file type, count, and size", progress=0.02)
        if not files:
            raise ValueError("No PDF files were supplied")
        if len(files) > settings.max_files_per_session:
            raise ValueError(f"A session supports at most {settings.max_files_per_session} PDFs")
        total_bytes = sum(len(data) for _, data in files)
        if total_bytes > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Combined upload exceeds {settings.max_upload_mb} MB")

        saved: list[tuple[str, bytes, Path]] = []
        seen_names: set[str] = set()
        for file_name, data in files:
            safe_name = Path(file_name).name
            if not safe_name.lower().endswith(".pdf"):
                raise ValueError(f"Unsupported file: {safe_name}")
            if safe_name in seen_names:
                stem = Path(safe_name).stem
                suffix = Path(safe_name).suffix
                counter = 2
                while f"{stem}_{counter}{suffix}" in seen_names:
                    counter += 1
                safe_name = f"{stem}_{counter}{suffix}"
            seen_names.add(safe_name)
            if not data.startswith(b"%PDF"):
                raise ValueError(f"{safe_name} does not appear to be a valid PDF")
            path = staging_uploads / safe_name
            path.write_bytes(data)
            saved.append((safe_name, data, path))

        all_pages: list[dict[str, Any]] = []
        page_counts: dict[str, int] = {}
        selectable_counts: dict[str, int] = {}
        jobs.update(job_id, step=1, phase="Extracting PDF text", detail="Reading text blocks in page order", progress=0.08)
        for file_index, (safe_name, _, path) in enumerate(saved):
            def page_progress(current: int, total: int, name: str) -> None:
                file_fraction = (file_index + current / max(total, 1)) / len(saved)
                jobs.update(
                    job_id,
                    step=1,
                    phase="Extracting PDF text",
                    detail=f"{name}: page {current}/{total}",
                    progress=0.08 + 0.22 * file_fraction,
                )

            pages = load_pdf(path, progress=page_progress)
            page_counts[safe_name] = len(pages)
            selectable_counts[safe_name] = sum(1 for page in pages if page.get("has_selectable_text"))
            all_pages.extend(pages)

        jobs.update(
            job_id,
            step=2,
            phase="Section-aware chunking",
            detail=f"Cleaning {len(all_pages)} pages and preserving headings",
            progress=0.34,
        )
        chunks = chunk_pages(all_pages)

        documents: list[dict[str, Any]] = []
        for safe_name, data, _ in saved:
            file_chunks = sum(1 for chunk in chunks if chunk["source"] == safe_name)
            total_pages = page_counts.get(safe_name, 0)
            selectable_pages = selectable_counts.get(safe_name, 0)
            warning = None
            if total_pages and selectable_pages / total_pages < 0.5:
                warning = "Many pages have little or no selectable text; OCR may improve coverage."
            documents.append(
                {
                    "file_name": safe_name,
                    "pages": total_pages,
                    "selectable_text_pages": selectable_pages,
                    "chunks": file_chunks,
                    "size_bytes": len(data),
                    "sha256": store.sha256(data),
                    "extraction_warning": warning,
                }
            )

        jobs.update(
            job_id,
            step=3,
            phase="Medical embedding",
            detail=f"Encoding {len(chunks)} chunks with the configured biomedical model",
            progress=0.42,
        )

        def embedding_progress(current: int, total: int) -> None:
            jobs.update(
                job_id,
                step=3,
                phase="Medical embedding",
                detail=f"Embedded {current}/{total} chunks",
                progress=0.42 + 0.38 * (current / max(total, 1)),
            )

        vector_store.index_staged(session_id, version, chunks, progress=embedding_progress)

        jobs.update(
            job_id,
            step=4,
            phase="Building hybrid retrieval",
            detail="Writing versioned evidence and BM25 indexes",
            progress=0.84,
        )
        store.write_version_payload(session_id, version, documents, chunks)
        target_upload_dir = store.version_upload_dir(session_id, version)
        shutil.rmtree(target_upload_dir, ignore_errors=True)
        target_upload_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_uploads, target_upload_dir)

        jobs.update(
            job_id,
            step=5,
            phase="Activating verified index",
            detail="Switching atomically to the new knowledge base",
            progress=0.94,
        )
        vector_store.activate(session_id, version)
        activated = True
        bm25_store.invalidate(session_id)
        store.cleanup_versions(session_id, keep_version=version)

        return {
            "session_id": session_id,
            "documents": documents,
            "pages": len(all_pages),
            "chunks": len(chunks),
            "embedding_model": settings.embedding_model if settings.embedding_backend == "sentence_transformers" else settings.ollama_embedding_model,
            "reranker_model": settings.reranker_model if settings.enable_reranker else None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception:
        if not activated:
            vector_store.discard_version(session_id, version)
            store.remove_version(session_id, version)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
