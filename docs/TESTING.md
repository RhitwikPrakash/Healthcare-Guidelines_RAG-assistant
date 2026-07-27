# Testing

## Backend tests inside Docker

```powershell
docker compose exec -T backend pytest -q
```

## Syntax verification

```powershell
docker compose exec -T backend python -m compileall app
```

## Public medical PDF retrieval test

This script downloads three official WHO guidelines from NCBI Bookshelf, checks PyMuPDF extraction, section-aware chunking and BM25 retrieval, and writes `public_pdf_test_report.json`:

```powershell
docker compose run --rm -v "${PWD}:/workspace" -w /workspace backend python scripts/test_public_pdfs.py
```

## Full end-to-end public PDF test

With the stack and Ollama models running:

```powershell
docker compose run --rm -v "${PWD}:/workspace" -w /workspace backend python scripts/test_public_pdfs.py --api http://backend:8000
```

The report records page counts, chunk counts, retrieval keyword hits, answer previews, citations and grounding confidence.
