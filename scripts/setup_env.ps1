param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$envExample = Join-Path $ProjectRoot ".env.example"
$envFile = Join-Path $ProjectRoot ".env"

if (-not (Test-Path $envExample)) {
    Write-Error ".env.example not found"
    exit 1
}

if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "Created .env from .env.example"
} else {
    Write-Host ".env already exists"
}

Write-Host "Edit $envFile and set ZVENOAI_API_KEY"
