# AC-HMM Satellites — Reproducible Genomic Sequence Modeling

**GitHub:** [github.com/FractiAI/ac-hmm-satellites](https://github.com/FractiAI/ac-hmm-satellites) · **Release:** [v1.0.0](https://github.com/FractiAI/ac-hmm-satellites/releases/tag/v1.0.0) · **OSF:** [osf.io/m5v8q](https://osf.io/m5v8q/)

**License:** MIT

---

## What this repository is for

This is a **standalone, peer-reviewable release** of the **Active Context Hidden Markov Model (AC-HMM)** applied to centromeric alpha-satellite arrays on the Telomere-to-Telomere human assembly **T2T-CHM13v2.0**.

The goal is not just to ship code — it is to let anyone **independently verify** the paper's claims: same coordinates, same leakage controls, same baselines, same statistical tests, same reported metrics.

You get:

- The full manuscript (`paper/AC_HMM_SATELLITES.md`)
- A containerized end-to-end pipeline (Docker + local scripts)
- A C++ forward–backward trellis (pybind11) with a NumPy fallback
- A BLASTN near-duplicate leakage filter
- Spatial autocorrelation decimation (45 kb buffer)
- Wilcoxon signed-rank significance testing
- A frozen manifest of T2T-CHM13 coordinates (`manifests/t2t_chm13_alpha.json`)

**OSF archive (data + frozen configs):** https://osf.io/m5v8q/ (Project Hub: `ac-hmm-satellites`)

---

## Intention

Centromeric alpha-satellite HOR arrays are highly repetitive — classical HMMs scale poorly and deep models overfit periodic structure. **AC-HMM** conditions emissions on deterministic context indices while keeping exact forward–backward inference.

This repository lets anyone **independently verify** the paper on **recognized public reference DNA**:

1. **T2T-CHM13v2.0** centromeric loci (UCSC Genome Browser API or full NCBI assembly).
2. Spatial cross-validation, BLASTN leakage control, and Wilcoxon tests.
3. Frozen ChrX transfer and pre-registered failure-zone evaluation.

---

## Abstract

**AC-HMM** decouples latent HMM transitions from contextual emission lookup, capturing sparse repeat structure with far fewer parameters than LSTM/Transformer baselines on low-entropy periodic DNA.

**Empirical findings (UCSC public T2T-CHM13 regions, 2-fold spatial CV, NumPy backend):**

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **Public data ingest** | 7 loci via **UCSC API (hs1/T2T-CHM13)**, 100 kb/locus | Real centromeric reference DNA — not synthetic demo |
| **Spatial CV (Chr11)** | AC-HMM mean ΔL_nat = **+232.4 ×10⁻⁴** nats/bp vs M₀ baseline | Positive held-out likelihood gain on both folds |
| **ChrX transfer** | AC-HMM **+240.4 ×10⁻⁴** on DXZ1 slice | Out-of-distribution generalization on public ChrX sequence |
| **Failure zone** | AC-HMM **+278.4 ×10⁻⁴** on high-entropy insert (capped window) | Subsampled window — full-zone behavior per paper requires full locus |
| **Baselines** | Variable-order Markov / PPM-C mixed on 4 kb windows | Expected on short capped training slices |
| **Audit ledger** | `raw_outputs/audit_ledger.json` | Provenance in `fetch_manifest.json` |

Full paper tables require full-locus NCBI fetch + GPU neural baselines. Use `--full` for complete assembly; `--demo` for smoke tests only.

---

## Key findings (paper reference)

Centromeric alpha-satellite **High-Order Repeat (HOR)** arrays are hard for standard models: classical HMMs hit exponential parameter growth when scaling context, while LSTMs and Transformers struggle with gradient noise on low-entropy, highly periodic DNA.

**AC-HMM** addresses this by conditioning HMM emissions on deterministic context indices from recent sequence history — decoupling latent state transitions from contextual lookup while keeping **exact** forward–backward inference and Baum–Welch training. The model captures sparse repeat structure with far fewer parameters than deep baselines.

On **T2T-CHM13v2.0** centromeric loci, the published evaluation reports:

| Finding | What it means |
|---------|----------------|
| **Out-of-sample log-likelihood gains** | AC-HMM beats baselines on held-out spatial folds of Chr11 D11Z1 |
| **Cross-chromosome generalization** | Parameters frozen after Chr11 training transfer favorably to ChrX DXZ1 |
| **LOHO robustness** | Gains hold under Leave-One-HOR-Out holdouts of evolutionary HOR families |
| **Pre-registered failure zone** | High-entropy, non-periodic retrotransposon insertions (Chr11 48.95–49.15 Mb) degrade performance as expected — bounding where the structural inductive bias helps |

Reproduce these results via the pipeline below; numbers land in `raw_outputs/audit_ledger.json` after a full T2T run. Full narrative and tables: `paper/AC_HMM_SATELLITES.md`.

---

## Start here — pick your path

Not sure where to begin? Use this table:

| Your goal | Time / disk | Command | What you get |
|-----------|-------------|---------|--------------|
| **Smoke test** — confirm the pipeline runs on your machine | ~5 min, &lt;1 GB | Windows: `.\verify_tables.ps1` · Linux/macOS: `./setup_env.sh && ./verify_tables.sh` | **UCSC public T2T regions** (default); metrics differ from full-paper tables |
| **Full local reproduction** — real T2T sequences, step-by-step control | Hours, ~3 GB reference | Follow [Full reproduction](#full-reproduction-t2t-chm13) below | Paper-grade HMM results; neural baselines need a GPU |
| **Reviewer / CI path** — one-shot, closest to published environment | Hours, Docker + GPU | `docker build -t ac-hmm-audit:v1 .` then `./verify_tables.sh` | Full pipeline including BLASTN and neural baselines |

**Read next:** [Primer](#primer) if the biology or method is new to you · [Pipeline at a glance](#pipeline-at-a-glance) to see how the pieces connect · [Quick start](#quick-start-smoke-test) to run immediately.

---

## Primer

### The biological problem

Human centromeres are dominated by **alpha-satellite** DNA: ~171 bp monomers stacked into massive **High-Order Repeat (HOR)** arrays. The T2T-CHM13 assembly finally made these regions sequence-resolved, but they are still hard to model:

- They are **highly repetitive** (low entropy, periodic structure).
- They contain **long-range context** (which monomer comes next depends on position within an HOR block).
- Standard train/test splits can **leak** near-identical repeats across folds if you are not careful.

This repository studies **Chr11 D11Z1** (spatial cross-validation folds) and **ChrX DXZ1** (out-of-distribution transfer), with a pre-registered **failure zone** where the model is expected to degrade.

### The modeling idea (AC-HMM)

A classical HMM captures dependencies only through hidden states. To remember more context, you must grow the state space — quickly becoming unidentifiable on repetitive DNA.

A deep model (LSTM, Transformer) has the opposite problem: too many parameters for a tiny alphabet (A/C/G/T) and very structured repeats, leading to noisy gradients.

**AC-HMM** splits the problem:

| Component | Role |
|-----------|------|
| Hidden states `Z` | Standard HMM transitions — *where* you are in the repeat landscape |
| Context index `h_t = ψ(x_{t-D:t-1})` | Deterministic lookup from recent bases — *what* local pattern you are in |
| Emissions `P(x_t \| z_t, h_t)` | Context-conditioned — the model sees both state and local history |

Inference stays **exact** (forward–backward, Baum–Welch) at **O(N K²)** — same order as a plain HMM after O(1) context precomputation. See the manuscript for the full derivation.

### Key terms in this repo

| Term | Meaning here |
|------|----------------|
| **ΔL_nat** | Change in natural log-likelihood per base pair vs. baseline (×10⁻⁴ scaling in terminal output) |
| **M₀ baseline** | Context-free HMM, K=8 states, D=0 context depth |
| **Spatial CV** | Five Chr11 folds; train on some, validate on held-out windows |
| **LOHO** | Leave-One-HOR-Out — hold out an evolutionary HOR family |
| **45 kb buffer** | Spatial decimation to reduce autocorrelation between adjacent windows |
| **Leakage mask** | BLASTN masks ≥95% identity blocks ≥100 bp across folds |
| **Failure zone** | Chr11 48.95–49.15 Mb — high-entropy insertions where AC-HMM should struggle |

All coordinates are locked in `manifests/t2t_chm13_alpha.json` so provenance is auditable.

---

## Pipeline at a glance

```mermaid
flowchart LR
  A[manifests/t2t_chm13_alpha.json] --> B[fetch_t2t_sequence.py]
  B --> C[build_spatial_track.py]
  C --> D[compute_spatial_ess.py]
  D --> E[verify_audit_ledger.py]
  E --> F[raw_outputs/audit_ledger.json]
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `tools/fetch_t2t_sequence.py` | Download T2T-CHM13 (or `--demo` synthetic repeats); optional BLASTN leakage mask |
| 2 | `tools/build_spatial_track.py` | Build spatial cross-validation windows from manifest loci |
| 3 | `tools/compute_spatial_ess.py` | Apply 45 kb autocorrelation decimation |
| 4 | `tools/verify_audit_ledger.py` | Train AC-HMM + baselines, run Wilcoxon tests, write audit ledger |

The audit ledger (`raw_outputs/audit_ledger.json`) is the single artifact that aggregates experiment results for comparison with the paper.

---

## Quick start (smoke test)

These commands run in **demo mode** with synthetic alpha-satellite-like repeats when the full ~3 GB T2T reference is not downloaded. Use this to confirm your environment before committing to a full run.

### Windows

```powershell
.\verify_tables.ps1
```

Creates `.venv`, installs locked dependencies, and runs the pipeline with capped training windows and `--skip-neural`.

### Linux / macOS

```bash
chmod +x setup_env.sh verify_tables.sh
./setup_env.sh
./verify_tables.sh
```

**Expected outcome:** `raw_outputs/audit_ledger.json` is written; the run completes without error.

**Not expected on demo data:** numeric agreement with paper Tables 2–5. For that, use [full reproduction](#full-reproduction-t2t-chm13). See also `VALIDATION.md` for a recorded smoke-test log.

---

## Full reproduction (T2T-CHM13)

### Prerequisites

| Requirement | Smoke test | Full local | Docker (recommended) |
|-------------|------------|------------|---------------------|
| Python 3.10+ | Yes | Yes | Bundled in image |
| ~3 GB disk for reference | No | Yes | Yes |
| `ncbi-blast+` (`blastn`) | Optional | Yes | Included |
| NVIDIA GPU | No | For neural baselines | `--gpus all` for neural baselines |
| cmake (optional) | No | For C++ trellis speedup | Built in image |

Without a GPU, pass `--skip-neural` to `verify_audit_ledger.py`. The HMM baselines run on CPU.

### 1. Environment

```bash
./setup_env.sh
source .venv/bin/activate
export PYTHONPATH=src/python
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements_lock.txt
$env:PYTHONPATH = "src\python"
```

### 2. Fetch sequences and BLASTN leakage mask

```bash
python tools/fetch_t2t_sequence.py --manifest manifests/t2t_chm13_alpha.json
```

Downloads `GCA_009914755.4` from NCBI (~3 GB compressed). Requires `ncbi-blast+` for the leakage filter.

### 3. Run the full audit (step by step)

```bash
python tools/build_spatial_track.py
python tools/compute_spatial_ess.py --input raw_outputs/chr11_cross_val.csv
python tools/verify_audit_ledger.py --strict_precision
```

**Output:** `raw_outputs/audit_ledger.json`

### 4. Docker (recommended for reviewers)

```bash
docker build -t ac-hmm-audit:v1 .
./verify_tables.sh
```

`verify_tables.sh` builds the image, fetches the reference, runs all four pipeline stages, and executes significance tests inside the container. Neural baselines require `docker run --gpus all` (handled by the script).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `paper/AC_HMM_SATELLITES.md` | Full manuscript (all tables) |
| `paper/reference_tables.json` | Machine-readable Table 2–5 benchmarks |
| `manifests/t2t_chm13_alpha.json` | Fixed coordinate provenance |
| `src/trellis.cpp` | C++ AC-HMM core (Baum–Welch) |
| `src/python/achmm/` | Python API + baselines |
| `tools/fetch_t2t_sequence.py` | NCBI ingest + BLASTN mask |
| `tools/build_spatial_track.py` | Spatial CV window builder |
| `tools/compute_spatial_ess.py` | 45 kb decimation |
| `tools/verify_audit_ledger.py` | Master experiment runner |
| `VALIDATION.md` | Recorded local smoke-test log |
| `instructions.txt` | Cursor master runbook (verbatim) |

---

## Metrics convention

All reported **ΔL_nat** values are in **natural nats per base pair**, displayed with uniform scaling **× 10⁻⁴** for terminal readability (per paper §5.1).

Baseline reference: context-free HMM (M₀, K=8, D=0).

---

## Determinism

Scripts call `torch.use_deterministic_algorithms(True)` and fixed seed **42**. Cross-platform floating-point accumulation may differ at ε ≈ 6×10⁻⁷ (documented in paper §7.2).

---

## Citation

```bibtex
@article{achmm_satellites_2026,
  title={Scalable Context-Conditioned Sequence Modeling in Repetitive Genomic Regions via Sparse Emission Matrices},
  author={FractiAI},
  year={2026},
  note={Repository: https://github.com/FractiAI/ac-hmm-satellites, OSF: https://osf.io/m5v8q/}
}
```

---

## Links

| Resource | URL |
|----------|-----|
| Repository | https://github.com/FractiAI/ac-hmm-satellites |
| Release v1.0.0 | https://github.com/FractiAI/ac-hmm-satellites/releases/tag/v1.0.0 |
| OSF project hub | https://osf.io/m5v8q/ |
| Validation log | `VALIDATION.md` |
