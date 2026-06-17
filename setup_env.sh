#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== INITIALIZING LOCAL WORKSPACE AND DIRECTORIES ==="
mkdir -p manifests checkpoints raw_outputs tools src data/reference data/sequences

echo "=== INITIALIZING PYTHON VIRTUAL ENVIRONMENT ==="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== INSTALLING FROZEN DEPENDENCIES ==="
pip install --upgrade pip
pip install -r requirements_lock.txt

echo "=== BUILDING C++ TRELLIS (optional) ==="
if command -v cmake &>/dev/null; then
  pip install -e . 2>/dev/null || python setup.py build_ext --inplace || echo "C++ build skipped"
else
  echo "cmake not found — using NumPy trellis fallback"
fi

echo "=== LOCAL ENVIRONMENT SETUP SUCCESSFULLY ==="
