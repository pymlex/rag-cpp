$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $Root "bin"

$mingwRoots = @(
    $env:MINGW_DLL_DIR,
    "C:\msys64\ucrt64\bin",
    "C:\msys64\mingw64\bin"
)
$sourceDir = $null
foreach ($candidate in $mingwRoots) {
    if ($candidate -and (Test-Path $candidate)) {
        $sourceDir = $candidate
        break
    }
}
if (-not $sourceDir) {
    Write-Host "MinGW bin directory not found. Set MINGW_DLL_DIR or install MSYS2 ucrt64."
    exit 1
}

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

$patterns = @(
    "libgcc_s*.dll",
    "libstdc++*.dll",
    "libwinpthread*.dll"
)
$copied = 0
foreach ($pattern in $patterns) {
    Get-ChildItem -Path $sourceDir -Filter $pattern -ErrorAction SilentlyContinue |
        ForEach-Object {
            Copy-Item $_.FullName -Destination (Join-Path $binDir $_.Name) -Force
            Write-Host "Copied $($_.Name)"
            $copied++
        }
}
if ($copied -eq 0) {
    Write-Host "No MinGW runtime DLLs copied from $sourceDir"
    exit 1
}
