Write-Host "=== RAG MCP Local system check ==="
Write-Host "Date:" (Get-Date -Format o)
Write-Host "OS:" ([System.Environment]::OSVersion.VersionString)
Write-Host "Arch:" $env:PROCESSOR_ARCHITECTURE
Write-Host "PowerShell:" $PSVersionTable.PSVersion
Write-Host ""
Write-Host "--- Tool versions ---"
$tools = @("python", "pip", "cmake", "gcc", "g++", "cl", "git")
foreach ($t in $tools) {
    $cmd = Get-Command $t -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "$t :" $cmd.Source
        if ($t -eq "python") { & python --version }
        if ($t -eq "pip") { & pip --version }
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
Write-Host "--- Python details ---"
if (Get-Command python -ErrorAction SilentlyContinue) {
    python -c "import sys,platform; print('executable=',sys.executable); print('version=',sys.version); print('platform=',platform.platform())"
    python -c "import struct; print('pointer_bits=', struct.calcsize('P')*8)"
}
Write-Host ""
Write-Host "--- MSVC environment (if cl exists) ---"
if (Get-Command cl -ErrorAction SilentlyContinue) {
    cl 2>&1 | Select-Object -First 5
}
Write-Host ""
Write-Host "Save this output and attach it if build fails."
