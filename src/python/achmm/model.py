"""Unified ACHMM facade — prefers C++ trellis, falls back to NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achmm.trellis_numpy import ACHMMNumpy, TrainResult as NumpyTrainResult

TRELLIS_BACKEND = "numpy"

try:
    import achmm_trellis  # type: ignore

    TRELLIS_BACKEND = "cpp"
except ImportError:
    achmm_trellis = None  # type: ignore


@dataclass
class TrainResult:
    log_likelihood: float
    iterations: int
    converged: bool
    flops: int
    backend: str


class ACHMM:
    def __init__(
        self,
        K: int = 8,
        D: int = 3,
        dirichlet_alpha: float = 10.0,
        occupancy_tau: float = 5.0,
        convergence_delta: float = 1e-6,
        max_iterations: int = 200,
        seed: int = 42,
    ):
        self.K = K
        self.D = D
        self.backend = TRELLIS_BACKEND
        if self.backend == "cpp":
            params = achmm_trellis.ACHMMParams()
            params.K = K
            params.D = D
            params.dirichlet_alpha = dirichlet_alpha
            params.occupancy_tau = occupancy_tau
            params.convergence_delta = convergence_delta
            params.max_iterations = max_iterations
            self._cpp = achmm_trellis.ACHMMTrellis(params)
            self._cpp.set_random_seed(seed)
        else:
            self._np = ACHMMNumpy(
                K=K,
                D=D,
                dirichlet_alpha=dirichlet_alpha,
                occupancy_tau=occupancy_tau,
                convergence_delta=convergence_delta,
                max_iterations=max_iterations,
                seed=seed,
            )

    def fit(self, symbols: np.ndarray, mask: np.ndarray) -> TrainResult:
        symbols = np.asarray(symbols, dtype=np.int32)
        mask = np.asarray(mask, dtype=np.uint8)
        if self.backend == "cpp":
            r = self._cpp.fit(symbols, mask)
            return TrainResult(r.log_likelihood, r.iterations, r.converged, r.flops, "cpp")
        r: NumpyTrainResult = self._np.fit(symbols, mask)
        return TrainResult(r.log_likelihood, r.iterations, r.converged, r.flops, "numpy")

    def score(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        symbols = np.asarray(symbols, dtype=np.int32)
        mask = np.asarray(mask, dtype=np.uint8)
        if self.backend == "cpp":
            return float(self._cpp.score(symbols, mask))
        return float(self._np.score(symbols, mask))

    def log_likelihood_per_bp(self, symbols: np.ndarray, mask: np.ndarray) -> float:
        ll = self.score(symbols, mask)
        active = int(mask.sum()) if mask.size else len(symbols)
        active = max(active, 1)
        return ll / active
