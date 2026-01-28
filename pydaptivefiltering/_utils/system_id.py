# pydaptivefiltering/_utils/system_id.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Data generation (System ID)
# =============================================================================

def generate_sign_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """MATLAB-like: x = sign(randn(K,1)). Returns float array shape (K,)."""
    x = np.sign(rng.standard_normal(K)).astype(float)
    # in case rng hits exact zero (rare), keep it deterministic: map 0 -> 1
    x[x == 0.0] = 1.0
    return x


def build_desired_from_fir(
    x: np.ndarray,
    Wo: np.ndarray,
    sigma_n2: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Desired signal for FIR unknown system:
      d[k] = Wo^T * X_k + n[k]
    where X_k is tapped delay line with length len(Wo), most recent first.

    Returns:
      d: (K,) float
      n: (K,) float
    """
    x = np.asarray(x, dtype=float).ravel()
    Wo = np.asarray(Wo, dtype=float).ravel()
    K = int(x.size)
    M = int(Wo.size - 1)

    n = (np.sqrt(float(sigma_n2)) * rng.standard_normal(K)).astype(float)

    x_pad = np.concatenate((np.zeros(M, dtype=float), x))
    d = np.zeros(K, dtype=float)

    # tapped delay line
    for k in range(K):
        Xk = x_pad[k : k + (M + 1)][::-1]
        d[k] = float(np.dot(Wo, Xk) + n[k])

    return d, n


# =============================================================================
# Coefficient history helper
# =============================================================================

def pack_theta_from_result(
    *,
    res,
    w_last: np.ndarray,
    n_coeffs: int,
    K: int,
) -> np.ndarray:
    """
    Return theta with shape (n_coeffs, K+1).

    Tries to use res.coefficients if available; otherwise fills only last column.
    Supports both (K+1, n_coeffs) and (n_coeffs, K+1).
    """
    theta = np.zeros((n_coeffs, K + 1), dtype=float)

    try:
        coeffs = np.asarray(res.coefficients)
        if coeffs.ndim == 2 and coeffs.shape == (K + 1, n_coeffs):
            theta[:, :] = coeffs.T
            return theta
        if coeffs.ndim == 2 and coeffs.shape == (n_coeffs, K + 1):
            theta[:, :] = coeffs
            return theta
    except Exception:
        pass

    theta[:, -1] = np.asarray(w_last, dtype=float).ravel()[:n_coeffs]
    return theta


# =============================================================================
# Progress
# =============================================================================

@dataclass
class ProgressConfig:
    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 50


def report_progress(
    *,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    mse_col: np.ndarray,
    cfg: ProgressConfig,
) -> None:
    """Print ensemble progress + tail MSE in dB."""
    if not cfg.verbose_progress:
        return

    if not (((l + 1) % cfg.print_every == 0) or ((l + 1) == ensemble)):
        return

    t_real1 = perf_counter()
    elapsed = t_real1 - t0
    avg_per = elapsed / (l + 1)
    eta = avg_per * (ensemble - (l + 1))

    tail = mse_col[max(0, len(mse_col) - cfg.tail_window) :]
    tail_db = 10.0 * np.log10(float(np.mean(tail)) + 1e-20)

    print(
        f"[Ensemble {l+1:>3}/{ensemble}] "
        f"time={(t_real1 - t_real0)*1e3:7.1f} ms | "
        f"tail_mse={tail_db:7.2f} dB | elapsed={elapsed:6.1f}s | ETA={eta:6.1f}s"
    )


# =============================================================================
# Plotting (single window)
# =============================================================================

def plot_system_id_single_figure(
    *,
    MSE_av: np.ndarray,
    MSEE_av: Optional[np.ndarray],
    MSEmin_av: Optional[np.ndarray],
    theta_av: Optional[np.ndarray],
    poles_order: int,
    title_prefix: str = "System ID",
) -> None:
    """
    One-window plot suggestion (2x2):
      (1) MSE
      (2) MSEmin (if provided) else blank
      (3) MSEE (aux) (if provided) else blank
      (4) coeff evolution (a1 and b0) if theta_av provided
    """
    import matplotlib.pyplot as plt  # local import avoids hard dependency in non-plot contexts

    K = int(len(MSE_av))
    k = np.arange(1, K + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (1) MSE
    ax1.plot(k, 10.0 * np.log10(np.asarray(MSE_av, dtype=float) + 1e-20))
    ax1.set_title(f"{title_prefix}: Learning Curve (MSE)")
    ax1.set_xlabel("iteration k")
    ax1.set_ylabel("MSE [dB]")
    ax1.grid(True)

    # (2) MSEmin
    if MSEmin_av is not None:
        ax2.plot(k, 10.0 * np.log10(np.asarray(MSEmin_av, dtype=float) + 1e-20))
        ax2.set_title(f"{title_prefix}: Learning Curve (MSEmin)")
        ax2.set_xlabel("iteration k")
        ax2.set_ylabel("MSEmin [dB]")
        ax2.grid(True)
    else:
        ax2.axis("off")

    # (3) MSEE (aux)
    if MSEE_av is not None:
        ax3.plot(k, 10.0 * np.log10(np.asarray(MSEE_av, dtype=float) + 1e-20))
        ax3.set_title(f"{title_prefix}: Learning Curve (aux / equation error)")
        ax3.set_xlabel("iteration k")
        ax3.set_ylabel("MSEE [dB]")
        ax3.grid(True)
    else:
        ax3.axis("off")

    # (4) coefficient evolution
    if theta_av is not None:
        th = np.asarray(theta_av, dtype=float)
        # expected shape: (n_coeffs, K+1)
        if th.ndim == 2 and th.shape[1] >= 2:
            # pick a1 (first pole coeff) and b0 (first numerator coeff)
            if poles_order > 0 and th.shape[0] > poles_order:
                a1 = th[0, 1:]             # skip k=0
                b0 = th[poles_order, 1:]   # first b is right after poles
                kk = np.arange(1, len(a1) + 1)
                ax4.plot(kk, a1, label="a1 (feedback)")
                ax4.plot(kk, b0, label="b0 (feedforward)")
                ax4.set_title(f"{title_prefix}: Coefficient evolution")
                ax4.set_xlabel("iteration k")
                ax4.set_ylabel("value")
                ax4.grid(True)
                ax4.legend()
            else:
                ax4.axis("off")
        else:
            ax4.axis("off")
    else:
        ax4.axis("off")

    fig.tight_layout()
    plt.show()