# Validation Run — Local Smoke Test (2026-06-17)

Environment: Windows 10, Python 3.12.10, NumPy trellis backend (C++ extension not built; MSVC redistributable absent). PyTorch neural baselines skipped (`--skip-neural`).

## Commands executed

```powershell
$env:PYTHONPATH = "src\python"
python tools\fetch_t2t_sequence.py --demo
python tools\build_spatial_track.py --max-bp 15000 --window 1000
python tools\compute_spatial_ess.py --input raw_outputs\chr11_cross_val.csv
python tools\verify_audit_ledger.py --max-train-bp 8000 --folds 2 --skip-neural --strict_precision
```

## Results summary

| Step | Status | Output |
|------|--------|--------|
| Fetch (demo) | PASS | `raw_outputs/fetch_manifest.json`, 7 loci × 80 kb synthetic repeats |
| Spatial track | PASS | `raw_outputs/chr11_cross_val.csv` (15 windows) |
| ESS decimation | PASS | `raw_outputs/decimated_blocks.csv` (1 block @ 45 kb buffer) |
| Audit ledger | PASS | `raw_outputs/audit_ledger.json` (253.8 s wall) |

## Interpretation

- **Demo sequences** are capped synthetic ACGT repeats — metrics will **not** match Table 2–5 in the paper (those require full T2T-CHM13 download + GPU neural baselines).
- **AC-HMM trellis** converges on NumPy backend; C++ backend builds inside Docker/Linux with cmake.
- **Failure zone** (random ACGT) shows expected degradation vs periodic folds — matches paper §7 limitation directionally.
- **BLASTN** skipped on this host (`blastn` not installed); Docker image includes `ncbi-blast+`.

## Full reproduction path

1. Linux/Docker: `./verify_tables.sh` (downloads T2T reference, runs BLASTN, GPU neural baselines)
2. Or stepwise per `README.md`

## Precision

`epsilon_stability: 6e-7` documented per paper §7.2.
