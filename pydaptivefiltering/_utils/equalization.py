# pydaptivefiltering/_utils/equalization.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple

import numpy as np


__all__ = [
    "toeplitz",
    "qam4_constellation_unit_var",
    "wgn_complex",
    "make_channel_matrix",
    "wiener_equalizer",
    "apply_channel",
    "generate_channel_data",
    "simulate_constellations",
    "best_phase_rotation",
    "hard_decision_qam4",
    "ser_qam4",
    "evm_mse",
    "cm_cost",
    "sign01",
    "wgn_real",
    "ProgressConfig",
    "report_progress",
    "apply_iir_channel",
    "tapped_delay_matrix",
]


def toeplitz(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Minimal Toeplitz (no SciPy)."""
    c, r = np.asarray(c), np.asarray(r)
    m, n = c.size, r.size
    out = np.empty((m, n), dtype=np.result_type(c, r))
    for i in range(m):
        for j in range(n):
            k = j - i
            out[i, j] = r[k] if k >= 0 else c[-k]
    return out


def qam4_constellation_unit_var() -> np.ndarray:
    """MATLAB: qammod(0:3,4)/sqrt(2) => {±1±j}/sqrt(2), unit average power."""
    return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2.0)


def wgn_complex(shape, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """Complex AWGN with E|n|^2 = sigma2 (each component variance sigma2/2)."""
    return (
        rng.normal(0.0, np.sqrt(sigma2 / 2.0), size=shape)
        + 1j * rng.normal(0.0, np.sqrt(sigma2 / 2.0), size=shape)
    )


def make_channel_matrix(H: np.ndarray, N: int) -> np.ndarray:
    """Toeplitz([H0, 0..], [H, 0..]) like MATLAB."""
    H = np.asarray(H, dtype=np.complex128).ravel()
    first_col = np.concatenate(([H[0]], np.zeros(N - 1, dtype=np.complex128)))
    first_row = np.concatenate((H, np.zeros(N - 1, dtype=np.complex128)))
    return toeplitz(first_col, first_row)


def wiener_equalizer(H: np.ndarray, N: int, sigma_x2: float, sigma_n2: float, delay: int) -> np.ndarray:
    """
    Matches MATLAB:
      HMatrix = toeplitz([H(1) zeros(1,N-1)],[H zeros(1,N-1)]);
      Rx = sigma_x2*eye(N+len(H)-1);
      Rn = sigma_n2*eye(N);
      Ry = HMatrix*Rx*HMatrix' + Rn;
      RxDeltaY = [zeros(1,delay) sigma_x2 zeros(1,N+len(H)-2-delay)]*(HMatrix');
      Wiener = (RxDeltaY*inv(Ry)).';
    Returns Wiener as shape (N,1).
    """
    HMatrix = make_channel_matrix(H, N)
    Rx = sigma_x2 * np.eye(N + len(H) - 1, dtype=np.complex128)
    Rn = sigma_n2 * np.eye(N, dtype=np.complex128)
    Ry = HMatrix @ Rx @ np.conj(HMatrix.T) + Rn

    rx_deltay_row = np.concatenate(
        (
            np.zeros(delay, dtype=np.complex128),
            np.array([sigma_x2], dtype=np.complex128),
            np.zeros(N + len(H) - 2 - delay, dtype=np.complex128),
        )
    )
    RxDeltaY = rx_deltay_row @ np.conj(HMatrix.T)  # (N,)
    return (RxDeltaY @ np.linalg.inv(Ry)).reshape(-1, 1)


def apply_channel(s: np.ndarray, H: np.ndarray, K: int) -> np.ndarray:
    """Streaming-style channel output length K."""
    return np.convolve(s, H, mode="full")[:K]


def generate_channel_data(
    rng: np.random.Generator,
    K: int,
    H: np.ndarray,
    sigma_n2: float,
    constellation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate s, x = channel(s)+noise, n (all length K)."""
    s = rng.choice(constellation, size=K)
    n = wgn_complex((K,), sigma_n2, rng)
    x = apply_channel(s, H, K) + n
    return s, x, n


@dataclass
class ProgressConfig:
    """
    Progress printing configuration for channel equalization ensemble loops.

    Parameters
    ----------
    verbose_progress : bool
        If True, prints progress.
    print_every : int
        Print every `print_every` realizations (also prints on the last one).
    tail_window : int
        Tail window (samples) used to compute tail MSE.
    optimize_verbose_first : bool
        Convenience flag used by examples to set optimize(verbose=True) for the first realization.
        Not used inside `report_progress`.
    """

    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 200
    optimize_verbose_first: bool = True


def report_progress(
    *,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    MSE_col: np.ndarray,
    K_eff: int,
    cfg: ProgressConfig,
    algo_tag: str = "Equalization",
    eps: float = 1e-300,
) -> None:
    """
    Print progress line compatible with the channel equalization examples.

    Expected call pattern:
        report_progress(l=l, ensemble=ensemble, t0=t0, t_real0=t_real0,
                        MSE_col=MSE[:, l], K_eff=K-delay, cfg=cfg)
    """
    if not cfg.verbose_progress:
        return

    is_last = (l + 1) >= ensemble

    if cfg.print_every <= 0:
        if not is_last:
            return
    else:
        if (not is_last) and (((l + 1) % int(cfg.print_every)) != 0):
            return

    mse_col = np.asarray(MSE_col, dtype=float).ravel()
    if mse_col.size == 0:
        tail_mse = float("nan")
    else:
        w = int(cfg.tail_window)
        if w <= 1:
            tail_mse = float(np.mean(mse_col[-min(mse_col.size, max(int(K_eff), 1)) :]))
        else:
            tail = mse_col[-min(mse_col.size, w) :]
            tail_mse = float(np.mean(tail))

    tail_db = 10.0 * float(np.log10(max(tail_mse, eps)))
    one_s = float(perf_counter() - t_real0)
    total_s = float(perf_counter() - t0)

    print(
        f"[{algo_tag}] {l+1:4d}/{ensemble} | tail MSE={tail_db:7.2f} dB | "
        f"one={one_s:6.2f}s | total={total_s:7.2f}s"
    )


def simulate_constellations(
    rng: np.random.Generator,
    H: np.ndarray,
    N: int,
    Ksim: int,
    sigma_n2: float,
    constellation: np.ndarray,
    w_final: np.ndarray,
    w_wiener: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      equalizerInputMatrix (N,Ksim),
      equalizerOutputVector (Ksim,),
      equalizerOutputVectorWiener (Ksim,)
    """
    HMatrix = make_channel_matrix(H, N)
    inputMatrix = rng.choice(constellation, size=(N + len(H) - 1, Ksim))
    noiseMatrix = wgn_complex((N, Ksim), sigma_n2, rng)
    equalizerInputMatrix = HMatrix @ inputMatrix + noiseMatrix

    w_final_v = np.asarray(w_final, dtype=np.complex128).reshape(-1)
    w_wiener_v = np.asarray(w_wiener, dtype=np.complex128).reshape(-1)

    y_eq = (np.conj(w_final_v) @ equalizerInputMatrix).ravel()
    y_w = (np.conj(w_wiener_v) @ equalizerInputMatrix).ravel()
    return equalizerInputMatrix, y_eq, y_w


def best_phase_rotation(y: np.ndarray, s_ref: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find phi minimizing || y*exp(-j phi) - s_ref || via LS phase."""
    phi = float(np.angle(np.vdot(s_ref, y)))
    y_rot = y * np.exp(-1j * phi)
    return y_rot, phi


def hard_decision_qam4(y: np.ndarray) -> np.ndarray:
    """Hard-decision for QAM4/QPSK points {±1±j}/sqrt(2)."""
    re = np.where(np.real(y) >= 0, 1.0, -1.0)
    im = np.where(np.imag(y) >= 0, 1.0, -1.0)
    return (re + 1j * im) / np.sqrt(2.0)


def ser_qam4(y_hat: np.ndarray, s_ref: np.ndarray) -> float:
    return float(np.mean(y_hat != s_ref))


def evm_mse(y_rot: np.ndarray, s_ref: np.ndarray) -> float:
    """Mean squared error after phase alignment (often called EVM^2)."""
    return float(np.mean(np.abs(y_rot - s_ref) ** 2))


def cm_cost(y: np.ndarray, R: float = 1.0) -> float:
    """Constant-modulus cost (useful for CMA/AP-CM outputs)."""
    return float(np.mean((np.abs(y) ** 2 - R) ** 2))


def sign01(x: np.ndarray) -> np.ndarray:
    """Real sign returning +/-1 (maps zeros to +1)."""
    y = np.sign(np.asarray(x)).astype(float)
    y[y == 0.0] = 1.0
    return y


def wgn_real(shape, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """Real AWGN with variance sigma2."""
    return rng.normal(0.0, np.sqrt(float(sigma2)), size=shape)


def apply_iir_channel(x: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Minimal real IIR filtering (Direct Form I-like), no SciPy.
    Implements: y[k] = sum_i num[i] x[k-i] - sum_{j>=1} den[j] y[k-j]
    Requires den[0] == 1.
    """
    x = np.asarray(x, dtype=float).ravel()
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    if den.size == 0 or den[0] != 1.0:
        raise ValueError("den[0] must be 1.0 for this minimal IIR implementation.")

    y = np.zeros_like(x, dtype=float)
    for k in range(x.size):
        acc = 0.0
        for i in range(num.size):
            if k - i >= 0:
                acc += num[i] * x[k - i]
        for j in range(1, den.size):
            if k - j >= 0:
                acc -= den[j] * y[k - j]
        y[k] = acc
    return y


def tapped_delay_matrix(x: np.ndarray, S: int) -> np.ndarray:
    """
    Tapped delay line matrix like MATLAB buffer(x,S,S-1,'nodelay'):
    returns shape (T, S) where T = len(x)-S+1, with most recent first.
    Row k = [x[k+S-1], ..., x[k]].
    """
    x = np.asarray(x).ravel()
    T = x.size - S + 1
    if T <= 0:
        raise ValueError(f"Need len(x) >= S. Got len(x)={x.size}, S={S}.")
    X = np.zeros((T, S), dtype=x.dtype)
    for k in range(T):
        X[k, :] = x[k : k + S][::-1]
    return X
