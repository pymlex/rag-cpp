param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$CorpusPath = ""
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $ProjectRoot "profiles\$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:RAG_CORPUS_PATH = $CorpusPath

$proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList "-m", "demo.gradio_app" `
    -WorkingDirectory (Join-Path $ProjectRoot "python") `
    -PassThru `
    -WindowStyle Hidden

Start-Sleep -Seconds 8
& "$ProjectRoot\.venv\Scripts\py-spy.exe" record -o (Join-Path $outDir "flamegraph.svg") --pid $proc.Id --duration 20 --format flamegraph
Stop-Process -Id $proc.Id -Force

$stats = @{
    pid = $proc.Id
    duration_sec = 20
    output = "flamegraph.svg"
}
$stats | ConvertTo-Json | Set-Content (Join-Path $outDir "profile_meta.json")
Write-Host "Profile saved to $outDir"
