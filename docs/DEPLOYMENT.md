# Deployment

## Local Docker deployment on Windows

From PowerShell in the extracted project folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Open:

- App: `http://localhost:8501`
- Backend health: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

The first setup downloads Docker images, the two Ollama models, PubMedBERT and the reranker. Later starts reuse persistent model volumes.

## Start and stop later

```powershell
docker compose up -d
docker compose down
```

Do not add `-v` to `docker compose down` unless you intentionally want to delete all downloaded models and indexed data.

## Cloudflare Tunnel

1. Create a Cloudflare Tunnel in the Zero Trust dashboard.
2. Add a public hostname whose service target is `http://frontend:8501`.
3. Copy the tunnel token.
4. Put it in `.env`:

```env
CLOUDFLARE_TUNNEL_TOKEN=your_token_here
```

5. Start the tunnel:

```powershell
.\scripts\start_cloudflare.ps1
```

Do not expose Ollama port `11434` publicly. Only the Streamlit frontend should be routed through the tunnel.

## Availability

The public URL works while the host machine, Docker Desktop and internet connection remain active. Disable sleep for the host when public access is required.

## Public-deployment security

The Docker services bind only to `127.0.0.1`. Public traffic should enter through the Cloudflare Tunnel, not through open router ports. Do not expose Ollama port `11434` or the backend port `8000` directly to the internet. For non-demo use, place Cloudflare Access authentication in front of the hostname.
