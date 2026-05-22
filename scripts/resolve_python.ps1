function Test-Python310Plus {
    param([string]$Exe)
    if (-not (Test-Path -LiteralPath $Exe)) {
        return $false
    }
    & $Exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

function Add-Candidate {
    param(
        [System.Collections.Generic.List[string]]$List,
        [hashtable]$Seen,
        [string]$Path
    )
    if (-not $Path) {
        return
    }
    $norm = $Path.Trim().Trim('"')
    if (-not $norm.EndsWith("python.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }
    if ($Seen.ContainsKey($norm)) {
        return
    }
    $Seen[$norm] = $true
    $List.Add($norm)
}

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"

$candidates = New-Object System.Collections.Generic.List[string]
$seen = @{}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    Add-Candidate -List $candidates -Seen $seen -Path $pythonCmd.Source
}

$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($pyCmd) {
    $defaultExe = & py -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1
    Add-Candidate -List $candidates -Seen $seen -Path $defaultExe

    $pyList = & py -0p 2>$null
    foreach ($row in $pyList) {
        if ($row -match '(python\.exe)\s*$') {
            $parts = $row -split '\s+'
            Add-Candidate -List $candidates -Seen $seen -Path $parts[-1]
        }
    }
}

$versionDirs = @("Python312", "Python311", "Python310")
foreach ($name in $versionDirs) {
    $exe = Join-Path $env:LOCALAPPDATA "Programs\Python\$name\python.exe"
    Add-Candidate -List $candidates -Seen $seen -Path $exe
}

foreach ($root in @("$env:LOCALAPPDATA\Programs\Python", "${env:ProgramFiles}\Python312", "${env:ProgramFiles}\Python311")) {
    if (-not (Test-Path $root)) {
        continue
    }
    Get-ChildItem -Path $root -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { Add-Candidate -List $candidates -Seen $seen -Path $_.FullName }
}

$ErrorActionPreference = $prevEap

foreach ($exe in $candidates) {
    if (Test-Python310Plus $exe) {
        Write-Output $exe
        exit 0
    }
}

Write-Error "Python 3.10+ is required. Install from https://www.python.org/downloads/ and ensure python is on PATH."
exit 1
