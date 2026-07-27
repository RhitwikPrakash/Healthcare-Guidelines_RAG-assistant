$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is not installed or docker is not on PATH."
}
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

docker compose up -d --build
Write-Host "Waiting for Ollama..."
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T ollama ollama list *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 2
}

docker compose exec -T ollama ollama pull medgemma1.5:4b
docker compose exec -T ollama ollama pull qwen3.5:4b

# Warm the medical embedding and reranker caches so the first upload is fast.
docker compose exec -T backend python prefetch_models.py

Write-Host ""
Write-Host "Healthcare RAG is ready at http://localhost:8501" -ForegroundColor Green
Write-Host "Backend API docs: http://localhost:8000/docs"
