param(
    [string]$BuildType = "Release",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "cpp\build"
$PythonDir = Join-Path $Root "python"

if (-not $PythonExe) {
    $PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    throw "Python not found at $PythonExe. Run scripts\install_all.ps1 first."
}

& $PythonExe -m pip install pybind11
$pybind11Dir = & $PythonExe -m pybind11 --cmakedir
if (-not $pybind11Dir) {
    throw "pybind11 cmake dir not found"
}

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

$generator = @()
if (Get-Command gcc -ErrorAction SilentlyContinue) {
    $generator = @("-G", "MinGW Makefiles")
    $ucrtBin = "C:\msys64\ucrt64\bin"
    if (Test-Path $ucrtBin) {
        $env:PATH = "$ucrtBin;$env:PATH"
    }
}

Push-Location $BuildDir
$cmakeArgs = @(
    ".."
    @($generator)
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DPython3_EXECUTABLE=$PythonExe"
    "-DCMAKE_PREFIX_PATH=$pybind11Dir"
)
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "cmake configure failed"
}
cmake --build . --config $BuildType
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "cmake build failed"
}
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
