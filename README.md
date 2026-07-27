# Healthcare Guidelines RAG Assistant — Super Local Edition

A local-first, Dockerised medical-document RAG assistant that researches one or many healthcare PDFs and produces evidence-grounded answers with page-level citations.

## Selected medical stack

- UI: Streamlit
- API and job orchestration: FastAPI
- Local inference: Ollama
- Primary medical model: MedGemma 1.5 4B
- Reasoning fallback: Qwen 3.5 4B
- Biomedical embeddings: PubMedBERT (`NeuML/pubmedbert-base-embeddings`)
- Sparse retrieval: BM25
- Dense vector store: ChromaDB
- Biomedical reranking: NCBI MedCPT cross-encoder
- Automatic reranker fallback: MiniLM cross-encoder
- PDF extraction: PyMuPDF
- Chunking: section-aware LangChain text splitting
- Deployment: Docker Compose and optional Cloudflare Tunnel

## What makes this version stronger

- No Gemini or Groq quota dependency during normal use.
- Biomedical embeddings instead of local feature hashing.
- Dense + BM25 hybrid retrieval with reciprocal-rank fusion.
- Cross-encoder reranking and neighbouring-evidence expansion.
- Adaptive focused, multi-hop and document-wide research strategies.
- Atomic versioned indexes: failed reprocessing does not erase the last working knowledge base.
- Live processing stages in the sidebar and answer area.
- Citation validation, evidence cards and grounding confidence.
- Primary and fallback local LLMs.
- Persistent Ollama, Hugging Face and RAG data volumes.
- Public access through Cloudflare without exposing Ollama.

## Windows quick start

Requirements: Windows 11, Docker Desktop, at least 16 GB RAM recommended, and internet for the first model downloads.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Open `http://localhost:8501`.

## Manual Docker start

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose exec -T ollama ollama pull medgemma1.5:4b
docker compose exec -T ollama ollama pull qwen3.5:4b
```

## Public URL with Cloudflare

Add your token to `.env`:

```env
CLOUDFLARE_TUNNEL_TOKEN=your_token_here
```

Then run:

```powershell
.\scripts\start_cloudflare.ps1
```

The host machine must remain powered on, awake, connected to the internet and running Docker Desktop.

## Project layout

```text
backend/        FastAPI, ingestion, retrieval, generation, validation and tests
frontend/       Streamlit interface and live pipeline display
scripts/        setup, verification, Cloudflare and public-PDF tests
docs/           architecture, deployment, profiles, commands and troubleshooting
cloudflare/     optional locally managed tunnel config example
docker-compose.yml
```

## Main flow

```text
Question -> adaptive query planning -> whole-PDF / multi-query research -> PubMedBERT + BM25 retrieval -> fusion -> MedCPT biomedical reranking -> evidence expansion -> MedGemma/Qwen answer -> citation and safety validation
```

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/DEPLOYMENT.md`
- `docs/MODEL_PROFILES.md`
- `docs/TERMINAL_COMMANDS.md`
- `docs/TESTING.md`
- `docs/TROUBLESHOOTING.md`

## Safety

This is an educational document assistant, not a diagnostic or prescribing system. It answers only from the uploaded evidence and should explicitly report insufficient or conflicting evidence. Always verify consequential decisions in the original guideline and with qualified clinical professionals.

## Model terms

The application code is licensed separately from downloaded models. Review the licence and acceptable-use terms shown by each model publisher before deployment, especially for MedGemma and other healthcare models.
