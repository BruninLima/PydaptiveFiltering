# ._utils/system_id.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple

import numpy as np

__all__ = [
    "generate_bpsk",
    "tapped_delay_response",
    "generate_system_id_data",
    "ProgressConfig",
    "report_progress",
]

def generate_bpsk(K: int, rng: np.random.Generator) -> np.ndarray:
    """MATLAB: x = sign(randn(K,1)) -> values in {-1, +1}."""
    x = np.sign(rng.standard_normal(K)).astype(float)
    x[x == 0] = 1.0
    return x

def tapped_delay_response(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Generate d_clean[k] = w^T X_k, with X_k = [x[k], x[k-1], ..., x[k-N+1]].
    Matches the MATLAB loop with a tapped delay line initialized at zeros.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    K = int(x.size)
    N = int(w.size)

    d = np.zeros(K, dtype=float)
    X = np.zeros(N, dtype=float)

    for k in range(K):
        X[1:] = X[:-1]
        X[0] = x[k]
        d[k] = float(np.dot(w, X))
    return d

def generate_system_id_data(
    rng: np.random.Generator,
    K: int,
    w0: np.ndarray,
    sigma_n2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, d, n) length K, with:
      x = sign(randn)
      d = w0^T X + n
      n ~ N(0, sigma_n2)
    """
    x = generate_bpsk(K, rng)
    n = np.sqrt(float(sigma_n2)) * rng.standard_normal(K)
    d_clean = tapped_delay_response(x, np.asarray(w0, dtype=float))
    d = d_clean + n
    return x, d, n

@dataclass
class ProgressConfig:
    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 50

def report_progress(
    *,
    algo_tag: str,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    mse_col: np.ndarray,
    cfg: ProgressConfig,
) -> None:
    """Print ensemble progress + tail MSE sanity metric."""
    if not cfg.verbose_progress:
        return
    if not (((l + 1) % cfg.print_every == 0) or ((l + 1) == ensemble)):
        return

    t1 = perf_counter()
    elapsed = t1 - t0
    avg = elapsed / (l + 1)
    eta = avg * (ensemble - (l + 1))

    mse_end_db = 10.0 * np.log10(float(mse_col[-1]) + 1e-20)
    tail = mse_col[max(0, len(mse_col) - cfg.tail_window):]
    tail_db = 10.0 * np.log10(float(np.mean(tail)) + 1e-20)

    print(
        f"[{algo_tag} {l+1:>3}/{ensemble}] "
        f"time={(t1 - t_real0)*1e3:7.1f} ms | "
        f"MSE_end={mse_end_db:7.2f} dB | tail={tail_db:7.2f} dB | "
        f"elapsed={elapsed:6.2f}s | ETA={eta:6.2f}s"
    )