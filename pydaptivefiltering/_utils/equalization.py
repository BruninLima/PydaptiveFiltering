# pydaptivefiltering/_utils/equalization.py
from __future__ import annotations

from typing import Tuple

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

    y_eq = np.conj(w_final).T @ equalizerInputMatrix
    y_w = np.conj(w_wiener.ravel()).T @ equalizerInputMatrix
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
