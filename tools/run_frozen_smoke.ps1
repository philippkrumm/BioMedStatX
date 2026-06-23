param(
    [string]$VenvScripts = "C:\bmx_venv\Scripts"
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$workPath = Join-Path $env:TEMP "BioMedStatX_build_smoke_$timestamp"
$distPath = Join-Path $env:TEMP "BioMedStatX_dist_smoke_$timestamp"
$reportPath = Join-Path $env:TEMP "BioMedStatX_smoke_$timestamp.txt"
$pyinstaller = Join-Path $VenvScripts "pyinstaller.exe"

if (-not (Test-Path -LiteralPath $pyinstaller)) {
    throw "PyInstaller not found: $pyinstaller"
}

Push-Location $projectRoot
try {
    & $pyinstaller BioMedStatX.spec --noconfirm --workpath $workPath --distpath $distPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $exePath = Join-Path $distPath "BioMedStatX\BioMedStatX.exe"
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Built executable not found: $exePath"
    }

    $env:BIOMEDSTATX_SMOKE_IMPORTS = "1"
    $env:BIOMEDSTATX_SMOKE_REPORT = $reportPath
    $process = Start-Process -FilePath $exePath -Wait -PassThru -WindowStyle Hidden

    if (Test-Path -LiteralPath $reportPath) {
        Get-Content -Path $reportPath
    }

    if ($process.ExitCode -ne 0) {
        throw "Frozen import smoke test failed with exit code $($process.ExitCode)"
    }

    Write-Host "Frozen import smoke test passed."
    Write-Host "Build output: $distPath"
}
finally {
    Remove-Item Env:\BIOMEDSTATX_SMOKE_IMPORTS -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOMEDSTATX_SMOKE_REPORT -ErrorAction SilentlyContinue
    Pop-Location
}
