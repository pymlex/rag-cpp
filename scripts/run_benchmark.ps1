$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Generate 20 QA with Zveno"
& $py (Join-Path $Root "benchmarks\generate_qa_dataset.py")

Write-Host "Run RAG evaluation"
& $py (Join-Path $Root "benchmarks\run_local_rag_eval.py")

Write-Host "Results in benchmarks\results\"
