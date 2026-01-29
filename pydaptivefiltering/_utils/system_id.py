# pydaptivefiltering/_utils/system_id.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple

import numpy as np


def generate_sign_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """Real sign(randn) like MATLAB."""
    return np.sign(rng.standard_normal(K)).astype(float)


def generate_qam4_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """4-QAM/QPSK with unit average power: {±1±j}/sqrt(2)."""
    re = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    im = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    return (re + 1j * im).astype(np.complex128) / np.sqrt(2.0)


def wgn_real(rng: np.random.Generator, shape, sigma_n2: float) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(sigma_n2), size=shape).astype(float)


def wgn_complex(rng: np.random.Generator, shape, sigma_n2: float) -> np.ndarray:
    return (
        rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=shape)
        + 1j * rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=shape)
    ).astype(np.complex128)


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


@dataclass
class ProgressConfig:
    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 50


def report_progress(
    algo_tag: str,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    mse_col: np.ndarray,
    cfg: ProgressConfig,
) -> None:
    if not cfg.verbose_progress:
        return
    if (l + 1) % cfg.print_every != 0 and (l + 1) != ensemble:
        return
    tail = mse_col[-cfg.tail_window:] if mse_col.size >= cfg.tail_window else mse_col
    tail_mean_db = 10.0 * np.log10(np.mean(tail) + 1e-12)
    dt_ens = perf_counter() - t0
    dt_one = perf_counter() - t_real0
    print(
        f"[{algo_tag}] {l+1:>4}/{ensemble} | "
        f"tail MSE={tail_mean_db:>7.2f} dB | "
        f"one={dt_one:>6.2f}s | total={dt_ens:>7.2f}s"
    )


def plot_system_id_single_figure(
    MSE_av: np.ndarray,
    MSEE_av: np.ndarray,
    MSEmin_av: np.ndarray,
    theta_av: np.ndarray,
    poles_order: int,
    title_prefix: str,
    show_complex_coeffs: bool,
) -> None:
    import matplotlib.pyplot as plt

    K = int(MSE_av.size)
    n_coeffs = int(theta_av.shape[0])

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.semilogy(MSE_av, label="MSE")
    plt.semilogy(MSEE_av, label="Aux MSE")
    plt.semilogy(MSEmin_av, label="Noise floor")
    plt.xlabel("iteration k")
    plt.ylabel("MSE")
    plt.title(f"{title_prefix} - Learning curves")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(n_coeffs):
        plt.plot(np.real(theta_av[i, :]), label=f"w[{i}]")
    plt.xlabel("k")
    plt.ylabel("real(theta)")
    plt.title(f"{title_prefix} - Coefficients (real)")
    plt.grid(True, alpha=0.3)

    if show_complex_coeffs:
        plt.subplot(2, 2, 3)
        for i in range(n_coeffs):
            plt.plot(np.imag(theta_av[i, :]), label=f"w[{i}]")
        plt.xlabel("k")
        plt.ylabel("imag(theta)")
        plt.title(f"{title_prefix} - Coefficients (imag)")
        plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(10 * np.log10(MSE_av + 1e-12))
    plt.xlabel("k")
    plt.ylabel("10log10(MSE)")
    plt.title(f"{title_prefix} - MSE (dB)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
