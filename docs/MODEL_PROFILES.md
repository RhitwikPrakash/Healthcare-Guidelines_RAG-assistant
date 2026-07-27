# Model profiles

## medical_quality — default

- Generator: `medgemma1.5:4b`
- Fallback and structured repair: `qwen3.5:4b`
- Dense embeddings: `NeuML/pubmedbert-base-embeddings`
- Sparse retrieval: BM25
- Primary reranker: `ncbi/MedCPT-Cross-Encoder`
- Automatic reranker fallback: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Best choice for a 16 GB RAM laptop or stronger machine.

## balanced

Set in `.env`:

```env
MODEL_PROFILE=balanced
LLM_MODEL=qwen3.5:4b
EMBEDDING_BACKEND=sentence_transformers
ENABLE_RERANKER=true
```

This keeps biomedical retrieval and uses the general reasoning model as primary generation.

## ultra_light

Set in `.env`:

```env
MODEL_PROFILE=ultra_light
LLM_MODEL=qwen3.5:4b
EMBEDDING_BACKEND=ollama
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
ENABLE_RERANKER=false
MAX_CONTEXT_CHARS=12000
FINAL_K=6
```

Then pull the embedding model:

```powershell

docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
```

Use this profile only when memory pressure makes the default medical profile impractical.

## NVIDIA GPU

Run Compose with the GPU override:

```powershell

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Docker Desktop, a supported NVIDIA driver and NVIDIA container GPU support are required.
