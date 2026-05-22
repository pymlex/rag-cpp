$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    throw ".env not found"
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*RAG_CORPUS_ROOT=(.+)$') {
        $corpus = $matches[1].Trim()
        $indexDir = Join-Path $corpus "rag_index"
        if (Test-Path $indexDir) {
            Remove-Item -Recurse -Force $indexDir
            Write-Host "Removed $indexDir"
        } else {
            Write-Host "No index at $indexDir"
        }
    }
}
