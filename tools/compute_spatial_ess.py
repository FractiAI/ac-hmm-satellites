#!/usr/bin/env python3
"""Spatial autocorrelation analysis and 45 Kb decimation for ESS correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = series - series.mean()
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    out = []
    for tau in range(max_lag + 1):
        if tau == 0:
            out.append(1.0)
        else:
            out.append(float(np.mean(x[:-tau] * x[tau:]) / var))
    return np.array(out)


def find_decimation_lag(
    series: np.ndarray, block_size: int = 1000, max_lag: int = 50000, M: int = 1000
) -> int:
    noise = 1.96 / np.sqrt(M)
    ac = autocorrelation(series, max_lag)
    for tau in range(1, len(ac)):
        if abs(ac[tau]) < noise:
            return max(tau * block_size, 1000)
    return 45000


def decimate_blocks(
    positions: np.ndarray,
    values: np.ndarray,
    buffer_bp: int,
    block_bp: int = 1000,
) -> pd.DataFrame:
    rows = []
    if len(positions) == 0:
        return pd.DataFrame(columns=["start", "end", "lambda_mean", "block_id"])
    start = int(positions[0])
    block_id = 0
    i = 0
    while i < len(positions):
        end = start + block_bp
        mask = (positions >= start) & (positions < end)
        if mask.any():
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "lambda_mean": float(np.mean(values[mask])),
                    "block_id": block_id,
                }
            )
            block_id += 1
        start = end + buffer_bp
        i = int(np.searchsorted(positions, start))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial ESS decimation")
    parser.add_argument("--input", required=True, help="CSV with position, delta_ll columns")
    parser.add_argument("--output", default=str(ROOT / "raw_outputs" / "decimated_blocks.csv"))
    parser.add_argument("--buffer-bp", type=int, default=45000)
    parser.add_argument("--block-bp", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "position" not in df.columns or "delta_ll" not in df.columns:
        raise SystemExit("Input CSV must contain position and delta_ll columns")

    positions = df["position"].to_numpy(dtype=np.int64)
    values = df["delta_ll"].to_numpy(dtype=np.float64)

    inferred_buffer = find_decimation_lag(values)
    buffer = args.buffer_bp if args.buffer_bp > 0 else inferred_buffer

    decimated = decimate_blocks(positions, values, buffer, args.block_bp)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decimated.to_csv(out_path, index=False)

    meta = {
        "input": args.input,
        "buffer_bp": buffer,
        "inferred_buffer_bp": int(inferred_buffer),
        "num_blocks": len(decimated),
        "autocorr_noise_limit": 1.96 / np.sqrt(1000),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Decimated {len(decimated)} independent blocks -> {out_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
