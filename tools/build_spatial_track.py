#!/usr/bin/env python3
"""Generate per-position delta log-likelihood track for spatial ESS analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from achmm.baselines import StandardHMM  # noqa: E402
from achmm.encoding import active_mask_from_sequence, encode_sequence  # noqa: E402
from achmm.model import ACHMM  # noqa: E402


def load_fasta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    seq = "".join(l for l in lines if not l.startswith(">"))
    return (
        np.array(encode_sequence(seq), dtype=np.int32),
        np.array(active_mask_from_sequence(seq), dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "t2t_chm13_alpha.json"))
    parser.add_argument("--output", default=str(ROOT / "raw_outputs" / "chr11_cross_val.csv"))
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--max-bp", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fetch_meta = ROOT / "raw_outputs" / "fetch_manifest.json"
    if fetch_meta.exists():
        fetched = json.loads(fetch_meta.read_text(encoding="utf-8")).get("loci", {})
    else:
        fetched = {}
    loci = manifest["loci"]
    fold_loci = [k for k, v in loci.items() if v.get("fold") is not None]

    def locus_path(lid: str) -> Path:
        if lid in fetched and "path" in fetched[lid]:
            return ROOT / fetched[lid]["path"]
        return ROOT / "data" / "sequences" / f"{lid}.fasta"

    all_sym, all_mask, positions = [], [], []
    offset = 0
    for lid in sorted(fold_loci, key=lambda x: loci[x]["fold"]):
        sym, mask = load_fasta(locus_path(lid))
        all_sym.append(sym)
        all_mask.append(mask)
        positions.extend(range(offset, offset + len(sym)))
        offset += len(sym)

    symbols = np.concatenate(all_sym)
    mask = np.concatenate(all_mask)
    positions = np.array(positions)
    if args.max_bp:
        symbols, mask, positions = symbols[: args.max_bp], mask[: args.max_bp], positions[: args.max_bp]

    defaults = manifest.get("model_defaults", {})
    hmm0 = StandardHMM(K=defaults.get("K", 8))
    hmm0.fit(symbols, mask)
    achmm = ACHMM(K=defaults.get("K", 8), D=defaults.get("D", 3))
    achmm.fit(symbols, mask)

    deltas = []
    w = args.window
    for i in range(0, len(symbols) - w, w):
        sl = slice(i, i + w)
        if mask[sl].sum() < w // 2:
            continue
        sub_sym, sub_m = symbols[sl], mask[sl]
        active = max(int(sub_m.sum()), 1)
        ll0 = hmm0.score(sub_sym, sub_m) / active
        lla = achmm.score(sub_sym, sub_m) / active
        deltas.append({"position": int(positions[i]), "delta_ll": float(lla - ll0)})

    df = pd.DataFrame(deltas)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} windows -> {out}")


if __name__ == "__main__":
    main()
