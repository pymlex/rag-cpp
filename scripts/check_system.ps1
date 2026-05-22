Write-Host "=== RAG MCP Local system check ==="
Write-Host "Date:" (Get-Date -Format o)
Write-Host "OS:" ([System.Environment]::OSVersion.VersionString)
Write-Host "Arch:" $env:PROCESSOR_ARCHITECTURE
Write-Host "PowerShell:" $PSVersionTable.PSVersion
Write-Host ""

$resolveScript = Join-Path $PSScriptRoot "resolve_python.ps1"
$py310 = $null
if (Test-Path $resolveScript) {
    $py310 = & $resolveScript 2>$null
    if ($LASTEXITCODE -eq 0 -and $py310) {
        Write-Host "Python 3.10+ for venv:" $py310
        & $py310 --version
    } else {
        Write-Host "Python 3.10+ for venv: NOT FOUND"
    }
}
Write-Host ""

Write-Host "--- Tool versions ---"
$tools = @("python", "pip", "cmake", "gcc", "g++", "cl", "git", "py")
foreach ($t in $tools) {
    $cmd = Get-Command $t -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "$t :" $cmd.Source
        if ($t -eq "python") { & python --version }
        if ($t -eq "pip") { & pip --version 2>$null }
        if ($t -eq "cmake") { & cmake --version | Select-Object -First 1 }
        if ($t -eq "gcc") { & gcc --version | Select-Object -First 1 }
        if ($t -eq "g++") { & g++ --version | Select-Object -First 1 }
        if ($t -eq "cl") { & cl 2>&1 | Select-Object -First 1 }
        if ($t -eq "git") { & git --version }
    } else {
        Write-Host "$t : NOT FOUND"
    }
}
Write-Host ""

Write-Host "--- Default python on PATH ---"
if (Get-Command python -ErrorAction SilentlyContinue) {
    python -c "import sys,platform; print('executable=',sys.executable); print('version=',sys.version); print('platform=',platform.platform())"
    python -c "import sys; print('ok_3.10_plus=', sys.version_info >= (3, 10))"
}
Write-Host ""
Write-Host "If ok_3.10_plus is False, install uses resolve_python.ps1 (py launcher or Python312 path)."
Write-Host "Save this output and attach it if build fails."
