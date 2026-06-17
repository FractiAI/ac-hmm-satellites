#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== STEP 1: CONSTRUCTING ENVIRONMENT CONTAINER ==="
docker build -t ac-hmm-audit:v1 .

echo "=== STEP 2: FETCHING TRACKS AND ENFORCING BLASTN PURGE ==="
docker run --rm -v "$ROOT":/workspace ac-hmm-audit:v1 \
    python3 tools/fetch_t2t_sequence.py --manifest manifests/t2t_chm13_alpha.json

echo "=== STEP 2b: BUILD SPATIAL TRACK ==="
docker run --rm -v "$ROOT":/workspace ac-hmm-audit:v1 \
    python3 tools/build_spatial_track.py --manifest manifests/t2t_chm13_alpha.json

echo "=== STEP 3: COMPUTING SPATIAL AUTOCORRELATION DECIMATION ==="
docker run --rm -v "$ROOT":/workspace ac-hmm-audit:v1 \
    python3 tools/compute_spatial_ess.py --input raw_outputs/chr11_cross_val.csv

echo "=== STEP 4: EXECUTING WILCOXON SIGNED-RANK SIGNIFICANCE TESTS ==="
docker run --rm -v "$ROOT":/workspace --gpus all ac-hmm-audit:v1 \
    python3 tools/verify_audit_ledger.py --strict_precision

echo "=== ARCHITECTURAL AUDIT COMPLETED SUCCESSFULLY ==="
