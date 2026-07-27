#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
grep -q '^CLOUDFLARE_TUNNEL_TOKEN=.' .env || { echo "Add CLOUDFLARE_TUNNEL_TOKEN to .env first." >&2; exit 1; }
docker compose --profile public up -d cloudflared
echo "Cloudflare Tunnel container started."
