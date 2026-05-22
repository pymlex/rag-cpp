param(
    [string]$ModelDir = "mlsa-iai-msu-lab/sci-rus-tiny",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$embDir = Join-Path $Root "third_party\embeddings-inference-server"
if (-not (Test-Path $embDir)) {
    throw "Embedding server not installed. Run scripts\install_all.ps1"
}
$env:MODEL_DIR = $ModelDir
Push-Location $embDir
& (Join-Path $Root ".venv\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port $Port
Pop-Location
