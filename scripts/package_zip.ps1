$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$zipPath = Join-Path $Root "rag_mcp_local_project.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$exclude = @("*.venv*", "*\cpp\build\*", "*\__pycache__\*", "*.pyc")
Compress-Archive -Path (Join-Path $Root "*") -DestinationPath $zipPath -Force
Write-Host "Created" $zipPath
