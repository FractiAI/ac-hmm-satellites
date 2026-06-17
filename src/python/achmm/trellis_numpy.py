"""Pure-Python / NumPy AC-HMM implementation (fallback when C++ extension unavailable)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG_ZERO = -1e300
MIN_PROB = 1e-12


def _safe_log(x: float) -> float:
    return math.log(max(float(x), MIN_PROB))


def _log_sum_exp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis)) + np.squeeze(amax, axis=axis)
    return float(out) if out.ndim == 0 else out


@dataclass
class TrainResult:
    log_likelihood: float
    iterations: int
    converged: bool
    flops: int


class ACHMMNumpy:
    def __init__(
        self,
        K: int = 8,
        D: int = 3,
        dirichlet_alpha: float = 10.0,
        occupancy_tau: float = 5.0,
        convergence_delta: float = 1e-6,
        max_iterations: int = 25,
        seed: int = 42,
    ):
        self.K = K
        self.D = D
        self.dirichlet_alpha = dirichlet_alpha
        self.occupancy_tau = occupancy_tau
        self.convergence_delta = convergence_delta
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        self.num_contexts = 4**D if D > 0 else 1
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 1.0 / K)
        self.B = np.full((K, self.num_contexts, 4), 1.0 / 4.0)
        self._context_idx: np.ndarray | None = None

    def _precompute_context(self, symbols: np.ndarray) -> np.ndarray:
        N = len(symbols)
        if self.D == 0:
            return np.zeros(N, dtype=np.int32)
        idx = np.zeros(N, dtype=np.int32)
        base = 4
        for t in range(self.D, N):
            window = symbols[t - self.D : t]
            if np.any(window < 0):
                continue
            val = 0
            for sym in window:
                val = val * base + int(sym)
            idx[t] = val
        return idx

    def _emission(self, symbols: np.ndarray, mask: np.ndarray, t: int, k: int) -> float:
        if not mask[t]:
            return 1.0
        sym = int(symbols[t])
        if sym < 0:
            return 1.0
        h = int(self._context_idx[t]) if self._context_idx is not None else 0
        return float(self.B[k, h, sym])

    def forward_backward(
        self, symbols: np.ndarray, mask: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, int]:
        N, K = len(symbols), self.K
        flops = 0
        log_alpha = np.full((N, K), LOG_ZERO)
        log_beta = np.full((N, K), LOG_ZERO)

        for k in range(K):
            emit = self._emission(symbols, mask, 0, k)
            log_alpha[0, k] = _safe_log(self.pi[k]) + _safe_log(emit)

        for t in range(1, N):
            for j in range(K):
                vals = log_alpha[t - 1] + np.log(np.maximum(self.A[:, j], MIN_PROB))
                log_alpha[t, j] = _log_sum_exp(vals) + _safe_log(
                    self._emission(symbols, mask, t, j)
                )
            flops += K * K

        log_ll = _log_sum_exp(log_alpha[N - 1])

        log_beta[N - 1] = 0.0
        for t in range(N - 2, -1, -1):
            for i in range(K):
                vals = []
                for j in range(K):
                    emit = self._emission(symbols, mask, t + 1, j)
                    vals.append(
                        _safe_log(self.A[i, j])
                        + _safe_log(emit)
                        + log_beta[t + 1, j]
                    )
                log_beta[t, i] = _log_sum_exp(np.array(vals))
            flops += K * K

        gamma = np.zeros((N, K))
        for t in range(N):
            unnorm = np.exp(log_alpha[t] + log_beta[t] - log_ll)
            gamma[t] = unnorm / unnorm.sum()

        xi = np.zeros((N - 1, K, K))
        for t in range(N - 1):
            log_xi = np.full((K, K), LOG_ZERO)
            for i in range(K):
                for j in range(K):
                    emit = self._emission(symbols, mask, t + 1, j)
                    log_xi[i, j] = (
                        log_alpha[t, i]
                        + _safe_log(self.A[i, j])
                        + _safe_log(emit)
                        + log_beta[t + 1, j]
                        - log_ll
                    )
            max_xi = np.max(log_xi)
            xi[t] = np.exp(log_xi - max_xi)
            xi[t] /= xi[t].sum()
            flops += K * K

        return float(log_ll), gamma, xi, flops

    def m_step(self, symbols: np.ndarray, mask: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        N, K, H = len(symbols), self.K, self.num_contexts
        alpha = self.dirichlet_alpha
        tau = self.occupancy_tau

        bg = np.zeros((K, 4))
        bg_den = np.zeros(K)
        active = mask.astype(bool)
        for t in np.where(active)[0]:
            a = int(symbols[t])
            if a < 0:
                continue
            bg[:, a] += gamma[t]
            bg_den += gamma[t]
        for k in range(K):
            bg[k] = (bg[k] + alpha) / (bg_den[k] + alpha * 4)

        self.A = xi.sum(axis=0)
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums[row_sums < MIN_PROB] = 1.0
        self.A /= row_sums

        emit_num = np.zeros((K, H, 4))
        emit_den = np.zeros((K, H))
        occ = np.zeros((K, H))
        for t in np.where(active)[0]:
            a = int(symbols[t])
            if a < 0:
                continue
            h = int(self._context_idx[t])
            emit_num[:, h, a] += gamma[t]
            emit_den[:, h] += gamma[t]
            occ[:, h] += gamma[t]

        for k in range(K):
            for h in range(H):
                w = min(1.0, occ[k, h] / tau)
                smoothed = (emit_num[k, h] + alpha) / (emit_den[k, h] + alpha * 4)
                self.B[k, h] = w * smoothed + (1.0 - w) * bg[k]
                self.B[k, h] /= self.B[k, h].sum()

        self.pi = gamma[0] / gamma[0].sum()

    def fit(self, symbols: np.ndarray, mask: np.ndarray) -> TrainResult:
        self._context_idx = self._precompute_context(symbols)
        prev = -np.inf
        total_flops = 0
        converged = False
        iterations = 0
        ll = 0.0
        for it in range(self.max_iterations):
            ll, gamma, xi, flops = self.forward_backward(symbols, mask)
            total_flops += flops
            self.m_step(symbols, mask, gamma, xi)
            iterations = it + 1
            if it > 0 and abs(ll - prev) < self.convergence_delta:
                converged = True
                break
            prev = ll
        return TrainResult(ll, iterations, converged, total_flops)

    def score(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        self._context_idx = self._precompute_context(symbols)
        ll, _, _, _ = self.forward_backward(symbols, mask)
        return ll
