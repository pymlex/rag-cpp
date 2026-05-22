$ErrorActionPreference = "Stop"

function Test-PythonVersion {
    param([string]$Exe)
    if (-not (Test-Path $Exe)) {
        return $false
    }
    $code = @"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
"@
    & $Exe -c $code
    return $LASTEXITCODE -eq 0
}

$candidates = New-Object System.Collections.Generic.List[string]

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($tag in @("3.12", "3.11", "3.10")) {
        $line = & py -$tag -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $line) {
            $candidates.Add($line.Trim())
        }
    }
}

$roots = @(
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:ProgramFiles\Python"
)
foreach ($root in $roots) {
    if (-not (Test-Path $root)) {
        continue
    }
    Get-ChildItem -Path $root -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { $candidates.Add($_.FullName) }
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    $candidates.Add((Get-Command python).Source)
}

$seen = @{}
foreach ($exe in $candidates) {
    if ($seen.ContainsKey($exe)) {
        continue
    }
    $seen[$exe] = $true
    if (Test-PythonVersion $exe) {
        Write-Output $exe
        exit 0
    }
}

Write-Error "Python 3.10+ is required. Install from https://www.python.org/downloads/ or run: py install 3.12"
exit 1
