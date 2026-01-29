# pydaptivefiltering/_utils/signal.py
from __future__ import annotations

import numpy as np

from typing import Tuple
from .typing import ArrayLike

__all__ = [
    "sign01",
    "tapped_delay_matrix",
    "apply_channel",
    "apply_iir_channel",
    "freq_response_fir",
    "freq_response_iir",
    "fir_filter_causal",
]

def sign01(x: np.ndarray) -> np.ndarray:
    """
    Real sign returning +/-1 (maps zeros to +1).

    Useful for BPSK-like decision-directed updates.
    """
    y = np.sign(np.asarray(x)).astype(float)
    y[y == 0.0] = 1.0
    return y

def tapped_delay_matrix(x: np.ndarray, S: int) -> np.ndarray:
    """
    Tapped delay line matrix like MATLAB buffer(x,S,S-1,'nodelay').

    Returns
    -------
    X : ndarray, shape (T, S)
        T = len(x) - S + 1
        Row k = [x[k+S-1], ..., x[k]]  (most recent first).
    """
    x = np.asarray(x).ravel()
    S = int(S)
    if S <= 0:
        raise ValueError(f"S must be positive. Got S={S}.")
    T = x.size - S + 1
    if T <= 0:
        raise ValueError(f"Need len(x) >= S. Got len(x)={x.size}, S={S}.")
    X = np.zeros((T, S), dtype=x.dtype)
    for k in range(T):
        X[k, :] = x[k : k + S][::-1]
    return X

def apply_channel(s: ArrayLike, H: ArrayLike, K: int) -> ArrayLike:
    """Streaming-style channel output length K."""
    return np.convolve(s, H, mode="full")[:K]

def apply_iir_channel(x: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Minimal real IIR filtering (no SciPy).
    Direct-form style:
        y[k] = sum_i num[i]*x[k-i] - sum_{j>=1} den[j]*y[k-j]
    Requires den[0] == 1.

    Notes
    -----
    This is intentionally minimal and intended for examples/utilities.
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

def freq_response_fir(
    b: np.ndarray,
    *,
    n_freq: int = 1024,
    whole: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FIR frequency response H(e^jw).

    Parameters
    ----------
    b : ndarray
        FIR coefficients in the usual convention:
            y[n] = sum_{m=0}^{M} b[m] x[n-m]
        (This matches your equalizer taps if you interpret w[0] as delay 0 tap.)
    n_freq : int
        Number of frequency samples.
    whole : bool
        If True, w spans [0, 2π). Else, w spans [0, π].

    Returns
    -------
    w : ndarray
        Frequencies in radians/sample.
    H : ndarray
        Complex frequency response sampled on w.
    """
    b = np.asarray(b).ravel().astype(np.complex128)

    w_max = 2.0 * np.pi if whole else np.pi
    w = np.linspace(0.0, w_max, int(n_freq), endpoint=False)

    m = np.arange(b.size)
    E = np.exp(-1j * np.outer(w, m)) 
    H = E @ b
    return w, H

def freq_response_iir(
    num: np.ndarray,
    den: np.ndarray,
    *,
    n_freq: int = 1024,
    whole: bool = False,
    eps: float = 1e-15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IIR frequency response H(e^jw) = B(e^jw)/A(e^jw), no SciPy.

    Implements polynomial evaluation:
        B(e^jw) = sum_k num[k] e^{-jwk}
        A(e^jw) = sum_k den[k] e^{-jwk}

    Requires den[0] != 0.

    Returns w in [0,π) (default) or [0,2π) (whole=True).
    """
    num = np.asarray(num).ravel().astype(np.complex128)
    den = np.asarray(den).ravel().astype(np.complex128)
    if den.size == 0 or den[0] == 0:
        raise ValueError("den must be non-empty and den[0] != 0.")

    w_max = 2.0 * np.pi if whole else np.pi
    w = np.linspace(0.0, w_max, int(n_freq), endpoint=False)

    k_num = np.arange(num.size)
    k_den = np.arange(den.size)

    E_num = np.exp(-1j * np.outer(w, k_num))
    E_den = np.exp(-1j * np.outer(w, k_den))

    B = E_num @ num
    A = E_den @ den
    A = np.where(np.abs(A) < eps, A + eps, A)

    H = B / A
    return w, H

def fir_filter_causal(h: ArrayLike, x: ArrayLike) -> ArrayLike:
    """Causal FIR filtering via convolution, equivalent to MATLAB's filter(h,1,x)."""
    h = np.asarray(h, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y_full = np.convolve(x, h, mode="full")
    return y_full[: x.size]


def align_by_xcorr_and_gain(
    y: np.ndarray,
    d: np.ndarray,
    max_lag: int = 256,
    remove_mean: bool = True,
    fit_gain: bool = True,
) -> dict:
    """
    Align y to d by maximizing cross-correlation over integer lag, optionally fit scalar gain.

    Returns dict:
      - lag: int (positive means y is delayed vs d; i.e., compare d[t] with y[t-lag])
      - gain: float (if fit_gain else 1.0)
      - y_aligned: ndarray
      - d_aligned: ndarray
    """
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    n = min(len(y), len(d))
    y = y[:n]
    d = d[:n]

    if remove_mean:
        y0 = y - np.mean(y)
        d0 = d - np.mean(d)
    else:
        y0, d0 = y, d

    max_lag = int(min(max_lag, max(0, n - 1)))

    best_lag = 0
    best_score = -np.inf
    eps = 1e-12

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            yy = y0[: n - lag]
            dd = d0[lag: n]
        else:
            yy = y0[-lag: n]
            dd = d0[: n + lag]

        m = min(len(yy), len(dd))
        if m < 8:
            continue
        yy = yy[:m]
        dd = dd[:m]

        num = float(np.dot(dd, yy))
        den = float(np.linalg.norm(dd) * np.linalg.norm(yy) + eps)
        score = num / den

        if score > best_score:
            best_score = score
            best_lag = lag

    # Build aligned segments
    lag = best_lag
    if lag >= 0:
        y_seg = y[: n - lag]
        d_seg = d[lag: n]
    else:
        y_seg = y[-lag: n]
        d_seg = d[: n + lag]

    # Fit scalar gain g minimizing ||d_seg - g*y_seg||^2
    if fit_gain:
        denom = float(np.dot(y_seg, y_seg) + 1e-12)
        gain = float(np.dot(d_seg, y_seg) / denom)
    else:
        gain = 1.0

    return {
        "lag": int(lag),
        "gain": float(gain),
        "y_aligned": gain * y_seg,
        "d_aligned": d_seg,
    }


def mse_aligned(y: np.ndarray, d: np.ndarray, max_lag: int = 256) -> tuple[float, int, float]:
    """Convenience: return (mse, lag, gain) after alignment."""
    out = align_by_xcorr_and_gain(y=y, d=d, max_lag=max_lag, remove_mean=True, fit_gain=True)
    e = out["d_aligned"] - out["y_aligned"]
    return float(np.mean(e**2)), out["lag"], out["gain"]

def check_pr_filterbank(hk: np.ndarray, fk: np.ndarray, L: int, N: int = 4096, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N).astype(float)

    M = hk.shape[0]
    # analysis + decimation
    xsb = []
    for m in range(M):
        xaux = np.convolve(x, hk[m], mode="full")[:N]  # aproxima teu fir_filter_causal
        xsb.append(xaux[::L])
    # synthesis
    y = np.zeros(N, dtype=float)
    for m in range(M):
        up = np.zeros(N, dtype=float)
        up[: len(xsb[m])*L : L] = xsb[m]
        y += np.convolve(up, fk[m], mode="full")[:N]

    mse0, lag, gain = mse_aligned(y, x, max_lag=256)
    return {"mse_aligned": mse0, "lag": lag, "gain": gain}