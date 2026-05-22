param(
    [switch]$SkipCpp
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "Step 1: system check"
& (Join-Path $Root "scripts\check_system.ps1")

Write-Host "Step 2: python venv"
$venv = Join-Path $Root ".venv"
if (-not (Test-Path $venv)) {
    python -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

Write-Host "Step 3: pip dependencies"
& $pip install --upgrade pip
& $pip install -r (Join-Path $Root "requirements.txt")

Write-Host "Step 4: .env"
$envExample = Join-Path $Root ".env.example"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "Created .env from .env.example. Fill ZVENOAI_API_KEY."
}

if (-not $SkipCpp) {
    Write-Host "Step 5: build C++ module (MinGW if gcc found)"
    & (Join-Path $Root "scripts\build_cpp.ps1")
}

Write-Host "Set RAG_CORPUS_ROOT in .env to your single corpus folder before daily use."

Write-Host "Step 6: install embedding server"
$embDir = Join-Path $Root "third_party\embeddings-inference-server"
if (-not (Test-Path $embDir)) {
    git clone https://github.com/pymlex/embeddings-inference-server.git $embDir
}
& $pip install -r (Join-Path $embDir "requirements.txt")

Write-Host "Installation finished."
Write-Host "Activate venv: .\.venv\Scripts\Activate.ps1"
Write-Host "Start embeddings: set MODEL_DIR=mlsa-iai-msu-lab/sci-rus-tiny; python third_party\embeddings-inference-server\run_server.py"
Write-Host "Start Gradio: python gradio_app.py"
