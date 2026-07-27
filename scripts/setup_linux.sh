#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
[ -f .env ] || cp .env.example .env

docker compose up -d --build
until docker compose exec -T ollama ollama list >/dev/null 2>&1; do sleep 2; done
docker compose exec -T ollama ollama pull medgemma1.5:4b
docker compose exec -T ollama ollama pull qwen3.5:4b

# Warm the medical embedding and reranker caches so the first upload is fast.
docker compose exec -T backend python prefetch_models.py

echo "Healthcare RAG is ready at http://localhost:8501"
echo "Backend API docs: http://localhost:8000/docs"
