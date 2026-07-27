# Verification report

**Build date:** 27 July 2026  
**Project:** Healthcare Guidelines RAG Assistant — Super Local Edition

## Completed verification

- Python syntax compilation completed for backend, frontend and scripts.
- Backend unit/API suite: **14/14 tests passed**.
- FastAPI smoke checks passed for `/`, `/health` and `/config`.
- Linux setup and Cloudflare shell scripts passed Bash syntax validation.
- Docker Compose, GPU override and Cloudflare YAML files parsed successfully.
- Versioned-index activation, rollback visibility and session cleanup were tested.
- Section headings were verified to carry across PDF page boundaries.
- Document-wide evidence sampling was verified across multiple PDFs and late sections.
- Citation ID cleanup, claim-level citation coverage and support-overlap checks were tested.
- Safety, query planning, PDF extraction and section-aware chunking tests passed.

## Authoritative public-PDF retrieval cases

The local extraction/chunking/BM25 verification script passed on three official WHO guideline PDFs distributed through NCBI Bookshelf:

| Document | Pages extracted | Chunks | Retrieval check |
|---|---:|---:|---|
| WHO guideline for pharmacological treatment of hypertension in adults | 61 | 199 | Passed; surfaced the treatment-initiation recommendation and threshold terms |
| WHO consolidated guidelines on tuberculosis: tuberculosis preventive treatment, second edition | 268 | 856 | Passed; surfaced preventive-treatment eligibility, HIV, household-contact and TB-disease evidence |
| WHO guidelines on physical activity and sedentary behaviour | 104 | 429 | Passed; surfaced adult activity duration, intensity and strengthening evidence |

Machine-readable details are saved in `public_pdf_test_report.json`. The test PDFs are intentionally excluded from the ZIP; `scripts/test_public_pdfs.py` downloads fresh copies when run.

## Environment limitation

The artifact-build environment did not provide Docker or an Ollama daemon, so the final Docker image build and live MedGemma/Qwen generation test could not be executed here. The project includes:

- Docker health checks and persistent volumes;
- automated model pulls and Hugging Face model prefetching;
- an end-to-end test mode (`scripts/test_public_pdfs.py --api ...`);
- runtime primary/fallback models and primary/fallback rerankers.

Run the Windows setup script on the target machine, then execute the Docker test commands in `docs/TESTING.md` to validate the exact local hardware/runtime combination.
