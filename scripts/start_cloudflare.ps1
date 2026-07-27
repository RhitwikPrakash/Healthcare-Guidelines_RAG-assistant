$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path ".env")) { throw "Create .env first." }
$tokenLine = Get-Content .env | Where-Object { $_ -match '^CLOUDFLARE_TUNNEL_TOKEN=.+' }
if (-not $tokenLine) { throw "Add CLOUDFLARE_TUNNEL_TOKEN to .env first." }
docker compose --profile public up -d cloudflared
Write-Host "Cloudflare Tunnel container started." -ForegroundColor Green
