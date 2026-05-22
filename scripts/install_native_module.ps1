$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "cpp\build"
$py = Join-Path $Root ".venv\Scripts\python.exe"

$artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $artifact) {
    $artifact = Get-ChildItem -Path $BuildDir -Recurse -Filter "ragdb_native*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $artifact) {
    Write-Host "No built module in cpp\build. Run .\scripts\build_cpp.ps1 first."
    exit 1
}

$sitePackages = & $py -c "import site; print(site.getsitepackages()[0])"
$targets = @(
    (Join-Path $Root $artifact.Name),
    (Join-Path $Root "python" $artifact.Name),
    (Join-Path $sitePackages $artifact.Name)
)
foreach ($dest in $targets) {
    Copy-Item $artifact.FullName -Destination $dest -Force
    Write-Host "Copied to $dest"
}

& $py -c "import ragdb_native; print('ragdb_native ok')"
