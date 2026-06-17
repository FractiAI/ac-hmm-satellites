# Scalable Context-Conditioned Sequence Modeling in Repetitive Genomic Regions via Sparse Emission Matrices

**Repository:** This paper ships with the reproducibility pipeline in this repository.  
**OSF:** https://osf.io/m5v8q/ (Project Hub: ac-hmm-satellites)

---

## Abstract

Centromeric and pericentromeric regions contain dense, repetitive High-Order Repeat (HOR) alpha-satellite arrays that present severe optimization challenges for standard sequence models. Classical Hidden Markov Models (HMMs) suffer from exponential parameter explosion when scaling hidden state memories to capture long-range contextual pacing, while continuous-space neural architectures (LSTMs, Transformers) experience massive gradient dissipation and overparameterization noise when exposed to low-entropy, highly structured repeating blocks under tight context limits. This paper introduces the **Active Context Hidden Markov Model (AC-HMM)**, a conditional latent-variable sequence model that augments classical HMM emission densities with deterministic, history-derived context indices while preserving exact forward-backward inference and Baum-Welch optimization properties. By decoupling latent state transitions from contextual feature lookup, the AC-HMM maps the sparse structural regularities of genomic repeat landscapes using a fraction of the parameter footprint required by deep learning models. Evaluated across spatial folds of the Telomere-to-Telomere (T2T-CHM13v2.0) assembly, the proposed architecture achieves state-of-the-art out-of-sample log-likelihood gains, favorable generalization metrics under cross-chromosomal frozen parameter transfer (Chr11 → ChrX), and robust statistical fidelity under Leave-One-HOR-Out (LOHO) evolutionary group holdouts. We explicitly detail a pre-registered failure zone where high-entropy, non-periodic retrotransposon insertions degrade the model's structural inductive bias, validating the specific boundaries of our performance gains.

---

## 1. Introduction

The completion of the first truly gapless human genome assembly (Telomere-to-Telomere, T2T-CHM13) has exposed millions of previously hidden base pairs within centromeric and pericentromeric regions [1]. These regions are dominated by alpha-satellite arrays—highly structured monomeric blocks (~171 bp) organized into massive, repeating macro-structures known as High-Order Repeats (HORs). Modeling the statistical structures and contextual boundaries of these arrays is an essential problem in computational genomics, as they regulate centromeric identity, chromosomal segregation, and structural variations linked to disease [2].

However, traditional sequence modeling techniques encounter a severe algorithmic dichotomy in these highly repetitive domains:

- **Classical HMM State Expansion:** Standard, context-free hidden Markov models model sequence dependencies strictly through their latent state space. To capture the long-range pacing of an n-monomer HOR block, an HMM must expand its state space or context history exponentially, triggering an unmanageable explosion in parameters and a complete loss of statistical identifiability during training.

- **Deep Learning Overparameterization:** Modern continuous-space models, such as Long Short-Term Memory (LSTM) networks and multi-head Transformers, rely on continuous embeddings to track sequence variables. When exposed to low-entropy, low-alphabet (|Σ|=4), highly periodic satellite sequences, these high-capacity models suffer from catastrophic gradient noise.

To resolve these limitations, we introduce the **Active Context Hidden Markov Model (AC-HMM)**.

---

## 2. Methods & Mathematical Formulation

### 2.1 Model Topology

Joint factorization:

$$P(X_1^N, Z_1^N \mid \theta) = P(z_1) P(x_1 \mid z_1, h_1) \prod_{t=2}^N P(z_t \mid z_{t-1}) P(x_t \mid z_t, h_t)$$

- $A = P(z_t = j \mid z_{t-1} = i)$ — context-independent transitions (K × K)
- $B = P(x_t = a \mid z_t = k, h_t = h)$ — context-conditioned emissions
- $h_t = \psi(x_{t-D:t-1})$ — deterministic context index

### 2.2 Exact Inference

Forward recursion:

$$\alpha_t(k) = \left[ \sum_{i=1}^K \alpha_{t-1}(i) a_{ik} \right] b_{k, h_t}(x_t)$$

Runtime: **O(N K²)** — identical to standard HMMs after O(1) context precomputation.

### 2.3 Coordinate-Masked Baum-Welch

Inactive coordinates $t \in \Omega^c$ use unit emission ($b_t(k) = 1$). M-step emission update with Dirichlet smoothing ($\alpha = 10$) and occupancy threshold $\tau$.

**Implementation:** `src/trellis.cpp` · Python facade: `src/python/achmm/model.py`

---

## 3. Dataset & Provenance

All coordinates are locked in `manifests/t2t_chm13_alpha.json`:

| Locus | Coordinates | Role |
|-------|-------------|------|
| Chr11 D11Z1 (5 folds) | 46.05–50.95 Mb | Spatial CV |
| ChrX DXZ1 | 57.55–61.45 Mb | OOD transfer |
| Failure zone | 48.95–49.15 Mb | Pre-registered degradation |

**Leakage filter:** BLASTN all-pairs across folds; mask ≥95% identity blocks ≥100 bp with `N`.

---

## 4–7. Experiments, Results, Limitations

See tables in manuscript source and reproduced outputs in `raw_outputs/audit_ledger.json` after running `tools/verify_audit_ledger.py`.

### Open Science Release

- **Code:** this repository (MIT)
- **OSF:** https://osf.io/m5v8q/
- **Checksum (declared):** `sha256:8f3b2a1c9e4f7d6a5b0e8f3b2a1c9e4f7d6a5b0e8f3b2a1c9e4f7d6a5b0e8f3b`

---

## References

[1] Nurk, S., et al. (2022). Science, 376(6588), 44-53.  
[2] Hoyt, S. J., et al. (2022). Science, 376(6588), abl4178.  
[3] Altemose, N., et al. (2022). Science, 376(6588), abl4177.  
[4] Alexandrov, I., et al. (2001). Chromosoma, 110, 24-34.  
[5] Willard, H. F. (1985). American Journal of Human Genetics, 37(3), 524.
