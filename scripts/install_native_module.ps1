$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "cpp\build"
$py = Join-Path $Root ".venv\Scripts\python.exe"
$sitePackages = Join-Path $Root ".venv\Lib\site-packages"

$artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $artifact) {
    $artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $artifact) {
    Write-Host "No built module in cpp\build. Run .\scripts\build_cpp.ps1 first."
    exit 1
}

& (Join-Path $Root "scripts\copy_mingw_runtime.ps1")

$pythonDir = Join-Path $Root "python"
$targets = @(
    (Join-Path $Root $artifact.Name),
    (Join-Path $pythonDir $artifact.Name),
    (Join-Path $sitePackages $artifact.Name)
)
foreach ($dest in $targets) {
    $parent = Split-Path $dest -Parent
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item $artifact.FullName -Destination $dest -Force
    Write-Host "Copied to $dest"
}

$ucrt = "C:\msys64\ucrt64\bin"
if (Test-Path $ucrt) {
    $env:PATH = "$ucrt;$env:PATH"
}
& $py -c "from python.native_index import NativeVectorIndex; print('ragdb_native ok')"
