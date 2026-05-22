param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
$target = Join-Path $ProjectRoot "third_party\embeddings-inference-server"
if (-not (Test-Path $target)) {
    git clone https://github.com/pymlex/embeddings-inference-server.git $target
}

$venv = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
& $venv install -r (Join-Path $target "requirements.txt")
Write-Host "Embeddings server cloned to $target"
Write-Host "Run: set MODEL_DIR=mlsa-iai-msu-lab/sci-rus-tiny && python run_server.py"
