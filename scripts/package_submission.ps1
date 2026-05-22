param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$zipPath = Join-Path $ProjectRoot "rag_mcp_local_submission.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

$items = @(
    "cpp",
    "python",
    "config",
    "benchmarks",
    "scripts",
    "uml",
    "main.py",
    "requirements.txt",
    ".env.example",
    "mcp.json",
    "README.md"
)

Compress-Archive -Path ($items | ForEach-Object { Join-Path $ProjectRoot $_ }) -DestinationPath $zipPath -Force
Write-Host "Created $zipPath"
