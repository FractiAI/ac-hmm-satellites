"""Baseline sequence models for AC-HMM comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

@dataclass
class BaselineResult:
    name: str
    log_likelihood: float
    log_likelihood_per_bp: float
    parameter_count: int
    flops: int
    wall_seconds: float


class StandardHMM:
    """Context-free HMM baseline (D=0)."""

    def __init__(self, K: int = 8, seed: int = 42):
        from achmm.model import ACHMM

        self.model = ACHMM(K=K, D=0, seed=seed)
        self.K = K

    def fit(self, symbols: np.ndarray, mask: np.ndarray) -> BaselineResult:
        import time

        t0 = time.perf_counter()
        r = self.model.fit(symbols, mask)
        ll = r.log_likelihood
        active = max(int(mask.sum()), 1)
        params = self.K * self.K + self.K * 4 + self.K
        return BaselineResult(
            "Standard HMM (M0)",
            ll,
            ll / active,
            params,
            r.flops,
            time.perf_counter() - t0,
        )

    def score(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        return self.model.score(symbols, mask)


class VariableOrderMarkov:
    """Pruned context tree (VOMM-style) with MDL penalty."""

    def __init__(self, d_max: int = 5, alpha_crit: float = 0.05):
        self.d_max = d_max
        self.alpha_crit = alpha_crit
        self.tree: dict = {}
        self.bg = np.full(4, 0.25)

    def _context_key(self, symbols: np.ndarray, t: int, d: int) -> tuple:
        return tuple(int(symbols[t - d + i]) for i in range(d) if t - d + i >= 0)

    def fit(self, symbols: np.ndarray, mask: np.ndarray) -> BaselineResult:
        import time

        t0 = time.perf_counter()
        counts: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
        total = np.zeros(4)
        active_idx = np.where(mask.astype(bool))[0]
        for t in active_idx:
            sym = int(symbols[t])
            if sym < 0:
                continue
            total[sym] += 1
            for d in range(1, self.d_max + 1):
                if t < d:
                    continue
                ctx = self._context_key(symbols, t, d)
                counts[ctx][sym] += 1

        self.bg = (total + 1.0) / (total.sum() + 4.0)
        self.tree = {}
        for ctx, cts in counts.items():
            if cts.sum() < 10:
                continue
            probs = (cts + 1.0) / (cts.sum() + 4.0)
            self.tree[ctx] = probs

        ll = self._loglik(symbols, mask)
        active = max(int(mask.sum()), 1)
        params = len(self.tree) * 4
        return BaselineResult(
            "Variable-Order Markov",
            ll,
            ll / active,
            params,
            len(active_idx) * self.d_max,
            time.perf_counter() - t0,
        )

    def _predict(self, symbols: np.ndarray, t: int) -> np.ndarray:
        for d in range(min(self.d_max, t), 0, -1):
            ctx = self._context_key(symbols, t, d)
            if ctx in self.tree:
                return self.tree[ctx]
        return self.bg

    def _loglik(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        ll = 0.0
        for t in np.where(mask.astype(bool))[0]:
            sym = int(symbols[t])
            if sym < 0:
                continue
            p = self._predict(symbols, t)[sym]
            ll += math.log(max(p, 1e-12))
        return ll

    def score(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        return self._loglik(symbols, mask)


class PPMC:
    """Prediction by Partial Matching (escape-C style)."""

    def __init__(self, D: int = 4):
        self.D = D
        self.counts: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(4))
        self.bg = np.full(4, 0.25)

    def fit(self, symbols: np.ndarray, mask: np.ndarray) -> BaselineResult:
        import time

        t0 = time.perf_counter()
        total = np.zeros(4)
        for t in np.where(mask.astype(bool))[0]:
            sym = int(symbols[t])
            if sym < 0:
                continue
            total[sym] += 1
            for d in range(0, min(self.D, t) + 1):
                ctx = tuple(int(symbols[t - d + i]) for i in range(d)) if d else ()
                self.counts[ctx][sym] += 1
        self.bg = (total + 1.0) / (total.sum() + 4.0)
        ll = self.score(symbols, mask)
        active = max(int(mask.sum()), 1)
        return BaselineResult(
            "PPM-C Compression",
            ll,
            ll / active,
            len(self.counts) * 4,
            int(mask.sum()) * self.D,
            time.perf_counter() - t0,
        )

    def _prob(self, symbols: np.ndarray, t: int, sym: int) -> float:
        for d in range(min(self.D, t), -1, -1):
            ctx = tuple(int(symbols[t - d + i]) for i in range(d)) if d else ()
            cts = self.counts.get(ctx)
            if cts is not None and cts.sum() > 0:
                return (cts[sym] + 1.0) / (cts.sum() + 4.0)
        return self.bg[sym]

    def score(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        ll = 0.0
        for t in np.where(mask.astype(bool))[0]:
            sym = int(symbols[t])
            if sym < 0:
                continue
            ll += math.log(max(self._prob(symbols, t, sym), 1e-12))
        return ll


class CharLSTM:
    def __init__(self, vocab: int = 5, embed: int = 16, hidden: int = 128, layers: int = 2):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, embed, padding_idx=4)
                self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, bidirectional=True)
                self.head = nn.Linear(hidden * 2, vocab)

            def forward(self, x):
                h, _ = self.lstm(self.embed(x))
                return self.head(h)

        self._factory = _Model

    def __call__(self):
        return self._factory()


class GenomicTransformer:
    def __init__(
        self,
        vocab: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        nlayers: int = 4,
        d_ff: int = 256,
    ):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, d_model, padding_idx=4)
                layer = nn.TransformerEncoderLayer(
                    d_model, nhead, d_ff, batch_first=True, dropout=0.0
                )
                self.encoder = nn.TransformerEncoder(layer, nlayers)
                self.head = nn.Linear(d_model, vocab)

            def forward(self, x):
                import torch

                causal = torch.nn.Transformer.generate_square_subsequent_mask(
                    x.size(1), device=x.device
                )
                h = self.encoder(self.embed(x), mask=causal, is_causal=True)
                return self.head(h)

        self._factory = _Model

    def __call__(self):
        return self._factory()


def _symbols_to_tokens(symbols: np.ndarray) -> np.ndarray:
    out = symbols.copy()
    out[out < 0] = 4
    return out.astype(np.int64)


def train_neural_baseline(
    name: str,
    model,
    symbols: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
) -> BaselineResult:
    import time

    import torch
    import torch.nn as nn

    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = tokens[:-1]
    y = tokens[1:]
    m = mask[1:].astype(bool)
    idx = np.where(m)[0]
    if len(idx) < batch_size:
        idx = np.arange(len(x))

    split = int(0.85 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 10) :]

    def batch_iter(indices):
        perm = indices.copy()
        np.random.shuffle(perm)
        for i in range(0, len(perm), batch_size):
            sl = perm[i : i + batch_size]
            yield (
                torch.tensor(x[sl], device=device).unsqueeze(0),
                torch.tensor(y[sl], device=device).unsqueeze(0),
            )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    stale = 0
    t0 = time.perf_counter()
    total_flops = 0
    param_count = sum(p.numel() for p in model.parameters())

    for _ in range(epochs):
        model.train()
        for xb, yb in batch_iter(train_idx):
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=4
            )
            loss.backward()
            opt.step()
            total_flops += param_count * xb.numel() * 6

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x[val_idx], device=device).unsqueeze(0)
            yv = torch.tensor(y[val_idx], device=device).unsqueeze(0)
            logits = model(xv)
            val_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yv.reshape(-1), ignore_index=4
            ).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, device=device).unsqueeze(0)
        logits = model(xt)[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        active_positions = np.where(mask[1:].astype(bool))[0]
        ll = 0.0
        for t in active_positions:
            target = int(y[t])
            if target == 4:
                continue
            ll += float(log_probs[t, target].item())

    active = max(int(mask.sum()), 1)
    return BaselineResult(
        name,
        ll,
        ll / active,
        param_count,
        total_flops,
        time.perf_counter() - t0,
    )
