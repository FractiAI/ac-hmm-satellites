#!/usr/bin/env python3
"""Compare audit_ledger.json outputs against paper reference tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare run results to paper reference tables")
    parser.add_argument(
        "--ledger",
        default=str(ROOT / "raw_outputs" / "audit_ledger.json"),
    )
    parser.add_argument(
        "--reference",
        default=str(ROOT / "paper" / "reference_tables.json"),
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))

    if not ledger_path.exists():
        print(f"No ledger at {ledger_path} — run verify_audit_ledger.py first.")
        return

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ref_t2 = {r["model"]: r for r in ref["table2_spatial_cv_chr11_d11z1"]["rows"]}

    print("=== Comparison: spatial CV (AC-HMM mean) ===")
    achmm_vals = []
    for fold_id, metrics in ledger.get("table2_spatial_cv", {}).items():
        v = metrics.get("AC-HMM (Proposed)")
        if v is not None:
            achmm_vals.append(v)
            print(f"  {fold_id}: run={v:+.4f}  (paper ref per-fold in reference_tables.json)")

    if achmm_vals:
        run_mean = sum(achmm_vals) / len(achmm_vals)
        paper_mean = ref_t2["AC-HMM (Proposed)"]["mean"]
        print(f"\n  Run mean (demo/full): {run_mean:+.4f}")
        print(f"  Paper mean (Table 2): {paper_mean:+.4f}")
        print("  Note: demo/subsampled runs will not match paper numbers.")

    print("\nReference tables loaded from:", args.reference)
    print("Backend:", ledger.get("backend"))
    print("Wall seconds:", ledger.get("wall_seconds"))


if __name__ == "__main__":
    main()
