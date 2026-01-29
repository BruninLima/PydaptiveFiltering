# pydaptivefiltering/_utils/system_id.py
from __future__ import annotations

from typing import Tuple
from .noise import wgn_complex, wgn_real
import numpy as np


def generate_sign_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """Real sign(randn) like MATLAB."""
    return np.sign(rng.standard_normal(K)).astype(float)


def generate_qam4_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """4-QAM/QPSK with unit average power: {±1±j}/sqrt(2)."""
    re = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    im = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    return (re + 1j * im).astype(np.complex128) / np.sqrt(2.0)

def build_desired_from_fir(
    x: np.ndarray,
    Wo: np.ndarray,
    sigma_n2: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    d = conv(x, Wo)[:K] + n
    Returns (d, n) with same length as x.
    """
    K = int(x.size)
    Wo = np.asarray(Wo)
    d_clean = np.convolve(x, Wo, mode="full")[:K]

    if np.iscomplexobj(d_clean) or np.iscomplexobj(Wo) or np.iscomplexobj(x):
        n = wgn_complex(rng, (K,), sigma_n2)
        d = d_clean.astype(np.complex128) + n
    else:
        n = wgn_real(rng, (K,), sigma_n2)
        d = d_clean.astype(float) + n

    return d, n


def _coeff_hist_to_array(coeff_hist, n_coeffs: int, K: int, dtype) -> np.ndarray:
    """
    Normalize coefficient history to shape (n_coeffs, K+1).
    Base class sometimes stores list of vectors, sometimes ndarray.
    """
    ch = np.asarray(coeff_hist)
    if ch.ndim == 2 and ch.shape[1] == n_coeffs:
        T = ch.shape[0]
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(T, K + 1)
        out[:, :T_use] = ch[:T_use, :].T
        return out

    if ch.ndim == 2 and ch.shape[0] == n_coeffs:
        T = ch.shape[1]
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(T, K + 1)
        out[:, :T_use] = ch[:, :T_use]
        return out

    try:
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(len(coeff_hist), K + 1)
        for t in range(T_use):
            out[:, t] = np.asarray(coeff_hist[t]).ravel()[:n_coeffs]
        return out
    except Exception as e:
        raise ValueError(f"Unsupported coefficient history format: {type(coeff_hist)}") from e


def pack_theta_from_result(res, w_last: np.ndarray, n_coeffs: int, K: int) -> np.ndarray:
    """
    Returns theta trajectory with shape (n_coeffs, K+1).
    Uses res.coefficients history if present; ensures last column = w_last.
    """
    dtype = np.result_type(w_last)
    theta = _coeff_hist_to_array(res.coefficients, n_coeffs=n_coeffs, K=K, dtype=dtype)
    theta[:, -1] = np.asarray(w_last).ravel()[:n_coeffs]
    return theta