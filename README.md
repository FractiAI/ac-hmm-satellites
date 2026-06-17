# AC-HMM Satellites — Reproducible Genomic Sequence Modeling

**GitHub:** [github.com/FractiAI/ac-hmm-satellites](https://github.com/FractiAI/ac-hmm-satellites) · **Release:** [v1.0.0](https://github.com/FractiAI/ac-hmm-satellites/releases/tag/v1.0.0) · **OSF:** [osf.io/m5v8q](https://osf.io/m5v8q/)

**Active Context Hidden Markov Model (AC-HMM)** for centromeric alpha-satellite arrays on T2T-CHM13v2.0.
This repository is a **standalone, peer-reviewable release** containing:

- The full paper (`paper/AC_HMM_SATELLITES.md`)
- Containerized pipeline (Docker + local scripts)
- C++ forward-backward trellis (pybind11) with NumPy fallback
- BLASTN near-duplicate leakage filter
- Spatial autocorrelation decimation (45 Kb buffer)
- Wilcoxon signed-rank significance testing
- Frozen manifest of T2T-CHM13 coordinates (`manifests/t2t_chm13_alpha.json`)

**OSF archive (data + frozen configs):** https://osf.io/m5v8q/ (Project Hub: `ac-hmm-satellites`)  
**License:** MIT
---

## Quick start (local validation)

### Windows

```powershell
.\verify_tables.ps1
```

### Linux / macOS

```bash
chmod +x setup_env.sh verify_tables.sh
./setup_env.sh
./verify_tables.sh
```

`verify_tables.ps1` runs in **demo mode** with synthetic alpha-satellite-like repeats when the full ~3 GB T2T reference is not yet downloaded. For the full empirical pipeline, use Docker or fetch the reference (see below).

---

## Full reproduction (T2T-CHM13)

### 1. Environment

```bash
./setup_env.sh
source .venv/bin/activate
export PYTHONPATH=src/python
```

### 2. Fetch sequences + BLASTN leakage mask

```bash
python tools/fetch_t2t_sequence.py --manifest manifests/t2t_chm13_alpha.json
```

Downloads `GCA_009914755.4` from NCBI (~3 GB compressed). Requires `ncbi-blast+` for leakage filtering.

### 3. Run full audit

```bash
python tools/build_spatial_track.py
python tools/compute_spatial_ess.py --input raw_outputs/chr11_cross_val.csv
python tools/verify_audit_ledger.py --strict_precision
```

Outputs: `raw_outputs/audit_ledger.json`

### 4. Docker (recommended for reviewers)

```bash
docker build -t ac-hmm-audit:v1 .
./verify_tables.sh
```

Requires NVIDIA GPU for neural baselines inside Docker (`--gpus all`). HMM baselines run on CPU.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `paper/AC_HMM_SATELLITES.md` | Manuscript |
| `manifests/t2t_chm13_alpha.json` | Fixed coordinate provenance |
| `src/trellis.cpp` | C++ AC-HMM core (Baum-Welch) |
| `src/python/achmm/` | Python API + baselines |
| `tools/fetch_t2t_sequence.py` | NCBI ingest + BLASTN mask |
| `tools/compute_spatial_ess.py` | 45 Kb decimation |
| `tools/verify_audit_ledger.py` | Master experiment runner |
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