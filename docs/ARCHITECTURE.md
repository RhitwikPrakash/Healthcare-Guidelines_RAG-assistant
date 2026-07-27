# Architecture

```text
Browser
  -> Streamlit frontend
  -> FastAPI job API
  -> PDF validation and PyMuPDF block extraction
  -> repeated-margin cleanup and section-aware chunking
  -> PubMedBERT dense embeddings
  -> ChromaDB persistent cosine index
  -> BM25 lexical index
  -> reciprocal-rank fusion
  -> NCBI MedCPT biomedical cross-encoder reranking
  -> MiniLM automatic fallback
  -> neighbouring passage expansion and section coverage
  -> adaptive focused / multi-hop / document-wide research
  -> MedGemma 1.5 4B through Ollama
  -> Qwen 3.5 4B fallback and citation repair
  -> citation validator, confidence estimate, safety notice
  -> grounded answer with source, page, section, and excerpt
```

## Reliability choices

- Versioned Chroma collections prevent a failed re-index from deleting the last working index.
- The active collection switches only after PDF extraction, chunking, embedding and persistence succeed.
- PDFs are isolated by browser session ID.
- Biomedical embeddings are used instead of feature hashing.
- Dense and lexical retrieval are fused before reranking.
- Complex questions can generate multiple retrieval paths.
- Whole-document questions add representative section coverage.
- Evidence IDs are validated before answers are displayed.
- The assistant refuses to fill evidence gaps with outside knowledge.

## Privacy

Normal operation is local. PDFs, chunks, Chroma files, Hugging Face models and Ollama models remain on the host machine. Cloudflare Tunnel forwards web traffic to the local Streamlit service; it does not move Ollama inference to Cloudflare.
