param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "cpp\build"
$PythonDir = Join-Path $Root "python"

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Push-Location $BuildDir
$generator = ""
if (Get-Command gcc -ErrorAction SilentlyContinue) {
    $generator = "-G MinGW Makefiles"
}
cmake .. $generator -DCMAKE_BUILD_TYPE=$BuildType
cmake --build . --config $BuildType
Pop-Location

$artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.pyd" | Select-Object -First 1
if (-not $artifact) {
    $artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.dll" | Select-Object -First 1
}
if (-not $artifact) {
    throw "ragdb_native module was not built"
}
Copy-Item $artifact.FullName -Destination (Join-Path $PythonDir $artifact.Name) -Force
Write-Host "Copied native module to" (Join-Path $PythonDir $artifact.Name)
