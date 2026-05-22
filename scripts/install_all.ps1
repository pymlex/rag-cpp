param(
    [switch]$SkipCpp,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "Step 1: system check"
& (Join-Path $Root "scripts\check_system.ps1")

if (-not $PythonExe) {
    $PythonExe = & (Join-Path $Root "scripts\resolve_python.ps1")
}
Write-Host "Using Python:" $PythonExe
& $PythonExe --version

Write-Host "Step 2: python venv"
$venv = Join-Path $Root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
$recreate = $false
if (Test-Path $venvPy) {
    $ok = & $venvPy -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Removing .venv created with Python older than 3.10"
        Remove-Item -Recurse -Force $venv
        $recreate = $true
    }
}
if (-not (Test-Path $venv)) {
    & $PythonExe -m venv $venv
}

$py = Join-Path $venv "Scripts\python.exe"

Write-Host "Step 3: pip dependencies"
& $py -m pip install --upgrade pip setuptools wheel
& $py -m pip install pybind11
& $py -m pip install -r (Join-Path $Root "requirements.txt")

$benchReq = Join-Path $Root "requirements-benchmark.txt"
if (Test-Path $benchReq) {
    Write-Host "Step 3b: optional tau2 benchmark (from GitHub)"
    & $py -m pip install -r $benchReq
}

Write-Host "Step 4: .env"
$envExample = Join-Path $Root ".env.example"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "Created .env from .env.example. Fill ZVENOAI_API_KEY."
}

if (-not $SkipCpp) {
    Write-Host "Step 5: build C++ module (MinGW if gcc found)"
    & (Join-Path $Root "scripts\build_cpp.ps1") -PythonExe $py
}

Write-Host "Set RAG_CORPUS_ROOT in .env to your single corpus folder before daily use."

Write-Host "Step 6: install embedding server"
$embDir = Join-Path $Root "third_party\embeddings-inference-server"
if (-not (Test-Path $embDir)) {
    git clone https://github.com/pymlex/embeddings-inference-server.git $embDir
}
& $py -m pip install -r (Join-Path $embDir "requirements.txt")

Write-Host "Installation finished."
Write-Host "Activate venv: .\.venv\Scripts\Activate.ps1"
Write-Host "Embeddings: .\scripts\run_embeddings_local.ps1"
Write-Host "Gradio: python gradio_app.py"
