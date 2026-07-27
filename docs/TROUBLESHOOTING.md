# Troubleshooting

## Ollama is offline

```powershell
docker compose ps
docker compose logs ollama --tail 100
docker compose restart ollama
```

## A model is missing

```powershell
docker compose exec -T ollama ollama pull medgemma1.5:4b
docker compose exec -T ollama ollama pull qwen3.5:4b
```

## First PDF processing is slow

The first run downloads PubMedBERT and the MedCPT biomedical cross-encoder into the persistent model cache. If MedCPT cannot load, the application automatically falls back to MiniLM. Later indexing reuses the cache.

## Laptop memory is too high

Use the `ultra_light` profile in `.env`, pull `qwen3-embedding:0.6b`, and restart:

```powershell
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
docker compose up -d --build
```

## Scanned PDF error

The default pipeline deliberately avoids automatic OCR because OCR can silently corrupt medical text. Use a searchable/text-based PDF or OCR it separately with a verified tool before uploading.

## Cloudflare URL does not open

```powershell
docker compose --profile public ps
docker compose logs cloudflared --tail 100
```

Confirm that the Cloudflare public hostname targets `http://frontend:8501` and that the host machine is awake.
