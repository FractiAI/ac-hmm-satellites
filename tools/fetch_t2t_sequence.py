#!/usr/bin/env python3
"""Fetch T2T-CHM13 alpha-satellite loci and apply BLASTN leakage masking."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from achmm.encoding import active_mask_from_sequence, encode_sequence  # noqa: E402

REF_DIR = ROOT / "data" / "reference"
SEQ_DIR = ROOT / "data" / "sequences"
META_OUT = ROOT / "raw_outputs" / "fetch_manifest.json"
UCSC_API = "https://api.genome.ucsc.edu/getData/sequence"
UCSC_GENOME = "hs1"  # T2T-CHM13v2.0 on UCSC


def fetch_ucsc_region(chrom: str, start: int, end: int, genome: str = UCSC_GENOME) -> str:
    """Fetch public T2T-CHM13 sequence slice via UCSC Genome Browser API."""
    import urllib.parse

    params = urllib.parse.urlencode(
        {"genome": genome, "chrom": chrom, "start": start, "end": end}
    )
    url = f"{UCSC_API}?{params}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "ac-hmm-satellites/1.0 (reproducible research)")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    dna = payload.get("dna", "").upper().replace("N", "")
    if not dna:
        raise RuntimeError(f"UCSC returned empty sequence for {chrom}:{start}-{end}")
    return dna


def download_reference(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    if not gz_path.exists():
        print(f"Downloading reference assembly: {url}")
        urllib.request.urlretrieve(url, gz_path)
    if not dest.exists():
        print(f"Decompressing {gz_path.name} ...")
        with gzip.open(gz_path, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return dest


def load_fasta_slice(fasta_path: Path, chrom: str, start: int, end: int) -> str:
    from pyfaidx import Fasta

    fa = Fasta(str(fasta_path), one_based_attributes=False, sequence_always_upper=True)
    key = chrom
    if key not in fa:
        for k in fa.keys():
            if k.endswith(chrom) or chrom in k:
                key = k
                break
        else:
            raise KeyError(f"Chromosome {chrom} not found in {fasta_path}")
    return str(fa[key][start:end])


def write_fasta(path: Path, locus_id: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f">{locus_id}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")


def run_blastn_mask(
    train_fastas: dict[str, Path],
    min_block: int,
    identity_threshold: float,
    blast_args: dict,
) -> dict[str, list[tuple[int, int]]]:
    """Return masked intervals per locus id from cross-fold BLASTN."""
    if shutil.which("blastn") is None:
        print("WARNING: blastn not found — skipping leakage filter (install ncbi-blast+).")
        return {k: [] for k in train_fastas}

    masked: dict[str, list[tuple[int, int]]] = {k: [] for k in train_fastas}
    ids = list(train_fastas.keys())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "all_loci"
        combined = tmp_path / "combined.fasta"
        with open(combined, "w", encoding="utf-8") as out:
            for lid, fpath in train_fastas.items():
                seq = open(fpath, encoding="utf-8").read().splitlines()
                seq = "".join(l for l in seq if not l.startswith(">"))
                out.write(f">{lid}\n{seq}\n")
        subprocess.run(
            ["makeblastdb", "-in", str(combined), "-dbtype", "nucl", "-out", str(db_path)],
            check=True,
            capture_output=True,
        )
        for i, qid in enumerate(ids):
            for sid in ids[i + 1 :]:
                qf = train_fastas[qid]
                out_file = tmp_path / f"{qid}_vs_{sid}.tsv"
                cmd = [
                    "blastn",
                    "-query",
                    str(qf),
                    "-db",
                    str(db_path),
                    "-outfmt",
                    "6 qseqid sseqid pident length qstart qend",
                    "-out",
                    str(out_file),
                    "-penalty",
                    str(blast_args.get("penalty", -3)),
                    "-reward",
                    str(blast_args.get("reward", 1)),
                    "-word_size",
                    str(blast_args.get("word_size", 11)),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                if not out_file.exists():
                    continue
                for line in out_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split("\t")
                    if len(parts) < 6:
                        continue
                    qseqid, sseqid, pident, length, qstart, qend = parts[:6]
                    if qseqid == sseqid:
                        continue
                    if float(pident) / 100.0 < identity_threshold:
                        continue
                    if int(length) < min_block:
                        continue
                    qs, qe = int(qstart) - 1, int(qend)
                    masked[qseqid].append((qs, qe))
    return masked


def apply_mask(seq: str, intervals: list[tuple[int, int]]) -> str:
    chars = list(seq)
    for start, end in intervals:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = "N"
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch T2T-CHM13 alpha-satellite loci")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "t2t_chm13_alpha.json"))
    parser.add_argument("--skip-download", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic smoke-test sequences only")
    mode.add_argument("--full", action="store_true", help="Full NCBI T2T assembly download (~3 GB)")
    mode.add_argument("--public", action="store_true", help="UCSC API region fetch (default)")
    args = parser.parse_args()
    use_demo = args.demo
    use_full = args.full

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    loci = manifest["loci"]
    leak_cfg = manifest.get("leakage_filter", {})

    extracted: dict[str, dict] = {}

    fetch_mode = "demo_synthetic"
    if use_demo:
        print("DEMO MODE: generating synthetic alpha-satellite-like repeats")
        unit = "ACGT" * 43  # ~171bp monomer proxy
        demo_cap = int(os.environ.get("ACHMM_DEMO_BP_CAP", "80000"))
        for lid, meta in loci.items():
            target_len = min(meta["end"] - meta["start"], demo_cap)
            blocks = target_len // len(unit) + 1
            seq = (unit * blocks)[:target_len]
            if lid == "failure_zone":
                import random

                random.seed(42)
                seq = "".join(random.choice("ACGT") for _ in seq)
            out_path = SEQ_DIR / f"{lid}.fasta"
            write_fasta(out_path, lid, seq)
            extracted[lid] = {
                **meta,
                "path": str(out_path.relative_to(ROOT)),
                "length_bp": len(seq),
                "active_bp": sum(1 for c in seq if c in "ACGT"),
            }
    elif use_full:
        fetch_mode = "ncbi_full_assembly"
        ref_path = REF_DIR / "GCA_009914755.4_genomic.fna"
        if not args.skip_download:
            download_reference(manifest["reference_url"], ref_path)
        for lid, meta in loci.items():
            print(f"Extracting {lid}: {meta['chromosome']}:{meta['start']}-{meta['end']}")
            seq = load_fasta_slice(ref_path, meta["chromosome"], meta["start"], meta["end"])
            out_path = SEQ_DIR / f"{lid}.fasta"
            write_fasta(out_path, lid, seq)
            extracted[lid] = {
                **meta,
                "path": str(out_path.relative_to(ROOT)),
                "length_bp": len(seq),
                "active_bp": sum(1 for c in seq if c in "ACGT"),
            }
    else:
        fetch_mode = "ucsc_public_api"
        region_cap = int(os.environ.get("ACHMM_REGION_BP_CAP", "100000"))
        print(f"PUBLIC MODE: UCSC T2T-CHM13 (hs1) region fetch (cap={region_cap} bp/locus)")
        for lid, meta in loci.items():
            span = min(meta["end"] - meta["start"], region_cap)
            end = meta["start"] + span
            print(f"Fetching {lid}: {meta['chromosome']}:{meta['start']}-{end}")
            seq = fetch_ucsc_region(meta["chromosome"], meta["start"], end)
            out_path = SEQ_DIR / f"{lid}.fasta"
            write_fasta(out_path, lid, seq)
            extracted[lid] = {
                **meta,
                "path": str(out_path.relative_to(ROOT)),
                "length_bp": len(seq),
                "active_bp": sum(1 for c in seq if c in "ACGT"),
                "fetch_end": end,
            }

    train_fastas = {
        lid: SEQ_DIR / f"{lid}.fasta"
        for lid, meta in loci.items()
        if meta.get("fold") is not None
    }
    masked_intervals = run_blastn_mask(
        train_fastas,
        leak_cfg.get("min_block_bp", 100),
        leak_cfg.get("identity_threshold", 0.95),
        leak_cfg.get("blastn", {}),
    )

    for lid, intervals in masked_intervals.items():
        if not intervals:
            continue
        fpath = SEQ_DIR / f"{lid}.fasta"
        lines = fpath.read_text(encoding="utf-8").splitlines()
        header, seq = lines[0], "".join(lines[1:])
        seq = apply_mask(seq, intervals)
        write_fasta(fpath, lid, seq)
        extracted[lid]["masked_intervals"] = intervals
        extracted[lid]["masked_bp"] = sum(e - s for s, e in intervals)

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(
        json.dumps(
            {
                "assembly": manifest["assembly"],
                "accession": manifest["accession"],
                "fetch_mode": fetch_mode,
                "data_source": (
                    "UCSC Genome Browser API (hs1/T2T-CHM13)"
                    if fetch_mode == "ucsc_public_api"
                    else manifest.get("reference_url", "NCBI")
                    if fetch_mode == "ncbi_full_assembly"
                    else "synthetic_demo"
                ),
                "loci": extracted,
                "leakage_filter_applied": bool(shutil.which("blastn")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {META_OUT}")
    print("Fetch complete.")


if __name__ == "__main__":
    main()
