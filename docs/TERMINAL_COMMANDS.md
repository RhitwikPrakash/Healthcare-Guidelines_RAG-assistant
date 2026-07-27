# Clean terminal commands

## First local setup

```powershell
cd "PATH_TO_EXTRACTED_FOLDER"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

## Verify

```powershell
python .\scripts\verify_stack.py
```

## Cloudflare public deployment

After adding `CLOUDFLARE_TUNNEL_TOKEN` to `.env`:

```powershell
.\scripts\start_cloudflare.ps1
```

## GitHub first push

```powershell
git init
git add .
git commit -m "Build local-first medical RAG assistant"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Later updates

```powershell
git add .
git commit -m "Improve medical RAG assistant"
git push
docker compose up -d --build
```

## Stop the local stack

```powershell
docker compose down
```

## Remove all local models and indexed data

Use only when you intentionally want a complete reset:

```powershell
docker compose down -v
```
