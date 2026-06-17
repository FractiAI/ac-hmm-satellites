#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== AC-HMM LOCAL VERIFICATION (Windows) ==="

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & .\.venv\Scripts\pip.exe install --upgrade pip
    & .\.venv\Scripts\pip.exe install -r requirements_lock.txt
}

$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "$Root\src\python"

Write-Host "=== FETCH (demo mode — synthetic repeats) ==="
& $py tools\fetch_t2t_sequence.py --manifest manifests\t2t_chm13_alpha.json --demo

Write-Host "=== BUILD SPATIAL TRACK ==="
& $py tools\build_spatial_track.py --manifest manifests\t2t_chm13_alpha.json --max-bp 100000

Write-Host "=== SPATIAL DECIMATION ==="
& $py tools\compute_spatial_ess.py --input raw_outputs\chr11_cross_val.csv

Write-Host "=== RUN EXPERIMENTS (capped bp for local validation) ==="
& $py tools\verify_audit_ledger.py --max-train-bp 8000 --folds 2 --skip-neural --strict_precision

Write-Host "=== VERIFICATION COMPLETE ==="
