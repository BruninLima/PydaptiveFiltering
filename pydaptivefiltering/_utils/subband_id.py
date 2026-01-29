# pydaptivefiltering/_utils/subband_id.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pydaptivefiltering._utils.system_id import (
    generate_sign_input,
    build_desired_from_fir,
    ProgressConfig,
    report_progress,
)


__all__ = [
    "SubbandIDConfig",
    "choose_subband_lengths",
    "run_subband_system_id",
    "plot_learning_curve",
]


@dataclass
class SubbandIDConfig:
    ensemble: int = 50
    K: int = 4096
    sigma_n2: float = 1e-3
    Wo: np.ndarray = np.array([0.32, -0.30, 0.50, 0.20], dtype=float)

    # progress
    progress: ProgressConfig = ProgressConfig(
        verbose_progress=True, print_every=10, tail_window=400
    )


def choose_subband_lengths(K: int, L: int) -> int:
    """
    Ensure K is a multiple of L (block advance). This avoids off-by-one slicing
    headaches in subband/block algorithms.
    """
    K = int(K)
    L = int(L)
    if L <= 0:
        raise ValueError("L must be > 0.")
    return (K // L) * L


def run_subband_system_id(
    *,
    make_filter: Callable[[], Any],
    L: int,
    cfg: SubbandIDConfig,
    seed: int = 0,
    verbose_first: bool = True,
) -> Dict[str, Any]:
    """
    Generic harness for subband/block algorithms that follow:
      res = flt.optimize(x_real, d_real)

    Parameters
    ----------
    make_filter:
        Callable that returns a *new* filter instance (e.g. lambda: pdf.CFDLMS(...)).
    L:
        Block advance (decimation) used by the algorithm (samples produced per iter).
        Used to choose K as multiple of L and to interpret output length.
    cfg:
        SubbandIDConfig with ensemble/K/sigma/Wo/progress.
    seed:
        Seed for reproducibility.
    verbose_first:
        Pass verbose=True only for first realization (helpful for CI logs).

    Returns
    -------
    dict with:
      - Wo
      - K_eff
      - MSE_av
      - MSEmin_av
      - MSE (K_eff, ensemble)
      - MSEmin (K_eff, ensemble)
      - total_time_s
    """
    rng_master = np.random.default_rng(int(seed))

    L = int(L)
    K = choose_subband_lengths(int(cfg.K), L)
    Wo = np.asarray(cfg.Wo, dtype=float).ravel()
    sigma_n2 = float(cfg.sigma_n2)
    ensemble = int(cfg.ensemble)
    progress_cfg = cfg.progress

    MSE = np.zeros((K, ensemble), dtype=float)
    MSEmin = np.zeros((K, ensemble), dtype=float)

    t0 = perf_counter()

    for l in range(ensemble):
        t_real0 = perf_counter()
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        x = generate_sign_input(rng, K).astype(float)
        d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)
        d = np.asarray(d, dtype=float).ravel()
        n = np.asarray(n, dtype=float).ravel()

        flt = make_filter()
        res = flt.optimize(x, d, verbose=(verbose_first and l == 0))

        e = np.asarray(res.errors).ravel().astype(float)
        K_eff = int(min(K, e.size))

        # store only effective region
        MSE[:K_eff, l] = e[:K_eff] ** 2
        MSEmin[:K_eff, l] = n[:K_eff] ** 2

        report_progress(
            algo_tag=type(flt).__name__,
            l=l, ensemble=ensemble, t0=t0, t_real0=t_real0,
            mse_col=MSE[:K_eff, l], cfg=progress_cfg,
        )

    total_time_s = float(perf_counter() - t0)
    K_eff_final = int(MSE.shape[0])

    MSE_av = np.mean(MSE, axis=1)
    MSEmin_av = np.mean(MSEmin, axis=1)

    return {
        "Wo": Wo,
        "K_eff": K_eff_final,
        "MSE": MSE,
        "MSEmin": MSEmin,
        "MSE_av": MSE_av,
        "MSEmin_av": MSEmin_av,
        "total_time_s": total_time_s,
    }


def plot_learning_curve(
    MSE_av: np.ndarray,
    MSEmin_av: Optional[np.ndarray] = None,
    title: str = "Learning curve",
) -> None:
    """
    Lightweight plot utility for subband examples (no theta plot).
    Import matplotlib lazily so tests can run headless without matplotlib if desired.
    """
    import matplotlib.pyplot as plt

    mse = np.asarray(MSE_av, dtype=float).ravel()
    x = np.arange(1, mse.size + 1)

    plt.figure(figsize=(10, 4))
    plt.semilogy(x, np.maximum(mse, 1e-20), label="MSE")
    if MSEmin_av is not None:
        msemin = np.asarray(MSEmin_av, dtype=float).ravel()
        plt.semilogy(x, np.maximum(msemin, 1e-20), label="MSEmin")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()
