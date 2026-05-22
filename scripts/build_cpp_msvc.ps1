param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
$buildDir = Join-Path $ProjectRoot "cpp\build_msvc"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Set-Location $buildDir

$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$pybindDir = & $pythonExe -m pybind11 --cmakedir

cmake .. -G "Visual Studio 17 2022" -A x64 -Dpybind11_DIR="$pybindDir"
cmake --build . --config Release -j

Copy-Item (Join-Path $buildDir "Release\rag_db.dll") (Join-Path $ProjectRoot "cpp\build\Release") -Force
Copy-Item (Join-Path $buildDir "Release\rag_db_native*.pyd") (Join-Path $ProjectRoot "python\rag_mcp") -Force
Write-Host "MSVC build complete."
