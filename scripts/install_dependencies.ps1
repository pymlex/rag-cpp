param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

python -m venv .venv
& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$ProjectRoot\.venv\Scripts\pip.exe" install -r requirements.txt
& "$ProjectRoot\.venv\Scripts\pip.exe" install pybind11 py-spy

& "$PSScriptRoot\setup_env.ps1" -ProjectRoot $ProjectRoot

Write-Host "Dependencies installed."
