# examples/example_systemID_stab_fast_rls.py
#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
# to estimate the filter coefficients of an unknown system given by Wo. In      #
# order to accomplish this task we use an adaptive filter with the same         #
# number of coefficients, N, as the unkown system. The procedure is:            #
# 1)  Excitate both filters (the unknown and the adaptive) with the signal      #
#   x. In this case, x is generated as sign(randn) (MATLAB example).             #
# 2)  Generate the desired signal, d = Wo' x + n, which is the output of the    #
#   unknown system considering some disturbance (noise) in the model. The       #
#   noise power is given by sigma_n2.                                           #
# 3)  Choose an adaptive filtering algorithm to govern the rules of coefficient #
#   updating.                                                                   #
#                                                                               #
#     Adaptive Algorithm used here: StabFastRLS                                 #
#                                                                               #
#################################################################################

from __future__ import annotations

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.system_id import (
    generate_system_id_data,
    ProgressConfig,
    report_progress,
)


def main(seed: int = 0, plot: bool = True):
    rng_master = np.random.default_rng(seed)

    # ----------------------------
    # Definitions (MATLAB)
    # ----------------------------
    ensemble = 100
    K = 500
    H = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], dtype=np.complex128)
    Wo = np.real(H).astype(float)   # unknown system (real)
    sigma_n2 = 0.04
    N = 4
    lambda_ = 0.97
    epsilon = 1.0

    if not hasattr(pdf, "StabFastRLS"):
        raise RuntimeError(
            "pydaptivefiltering does not expose 'StabFastRLS'. "
            "Check the export name (e.g., StabFastRLS, StabilizedFastRLS, etc.)."
        )

    # ----------------------------
    # Memory allocation (MATLAB-like)
    # ----------------------------
    W = np.zeros((N, K + 1, ensemble), dtype=float)   # REAL coefficients
    MSE = np.zeros((K, ensemble), dtype=float)
    MSEPost = np.zeros((K, ensemble), dtype=float)
    MSEmin = np.zeros((K, ensemble), dtype=float)

    cfg = ProgressConfig(verbose_progress=True, print_every=10, tail_window=50)

    t0 = perf_counter()

    for l in range(ensemble):
        t_real0 = perf_counter()

        # independent seed per realization
        seed_l = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed_l)

        # IMPORTANT: generate REAL x and d (StabFastRLS requires real signals)
        x, d, n = generate_system_id_data(rng=rng, K=K, w0=Wo, sigma_n2=sigma_n2)
        x = np.asarray(x, dtype=float).ravel()
        d = np.asarray(d, dtype=float).ravel()

        flt = pdf.StabFastRLS(
            filter_order=(N - 1),
            forgetting_factor=lambda_,
            epsilon=epsilon,
        )

        # print internal runtime only for first run
        res = flt.optimize(x, d, verbose=(l == 0), return_internal_states=False)

        e_ap = np.asarray(res.errors).ravel()
        e_post = np.asarray(res.extra.get("errors_posteriori", res.errors)).ravel()

        MSE[:, l] = np.abs(e_ap) ** 2
        MSEPost[:, l] = np.abs(e_post) ** 2
        MSEmin[:, l] = np.abs(n) ** 2

        # store coefficient history if available; otherwise final weight
        try:
            coeffs = np.asarray(res.coefficients)
            # Common patterns:
            # (K+1, N) or (N, K+1)
            if coeffs.ndim == 2 and coeffs.shape == (K + 1, N):
                W[:, :, l] = coeffs.T
            elif coeffs.ndim == 2 and coeffs.shape == (N, K + 1):
                W[:, :, l] = coeffs
            else:
                W[:, -1, l] = np.asarray(flt.w, dtype=float).ravel()
        except Exception:
            W[:, -1, l] = np.asarray(flt.w, dtype=float).ravel()

        report_progress(
            algo_tag="StabFastRLS",
            l=l,
            ensemble=ensemble,
            t0=t0,
            t_real0=t_real0,
            mse_col=MSE[:, l],
            cfg=cfg,
        )

    total_time = perf_counter() - t0
    print(f"[Example/StabFastRLS] Total ensemble time: {total_time:.2f} s ({total_time/ensemble:.3f} s/realization)")

    # ----------------------------
    # Averaging (MATLAB)
    # ----------------------------
    W_av = np.mean(W, axis=2)            # (N, K+1)
    MSE_av = np.mean(MSE, axis=1)        # (K,)
    MSEPost_av = np.mean(MSEPost, axis=1)
    MSEmin_av = np.mean(MSEmin, axis=1)

    # ----------------------------
    # Plots
    # ----------------------------
    if plot:
        k = np.arange(1, K + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(k, 10 * np.log10(MSE_av + 1e-20), "-k")
        axes[0].set_title("Learning Curve for MSE")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("MSE [dB]")
        axes[0].grid(True)

        axes[1].plot(k, 10 * np.log10(MSEPost_av + 1e-20), "-k")
        axes[1].set_title("Learning Curve for MSE (a posteriori)")
        axes[1].set_xlabel("k"); axes[1].set_ylabel("MSEPost [dB]")
        axes[1].grid(True)

        axes[2].plot(k, 10 * np.log10(MSEmin_av + 1e-20), "-k")
        axes[2].set_title("Learning Curve for MSEmin")
        axes[2].set_xlabel("k"); axes[2].set_ylabel("MSEmin [dB]")
        axes[2].grid(True)

        fig.tight_layout()

        fig2, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(W_av[0, :])
        ax[0].set_title("Evolution of the 1st coefficient")
        ax[0].set_ylabel("Coefficient")
        ax[0].grid(True)

        ax[1].plot(W_av[1, :] if N > 1 else W_av[0, :])
        ax[1].set_title("Evolution of the 2nd coefficient (for reference)")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("Coefficient")
        ax[1].grid(True)

        fig2.tight_layout()
        plt.show()

    return {
        "Wo": Wo,
        "W_av": W_av,
        "MSE_av": MSE_av,
        "MSEPost_av": MSEPost_av,
        "MSEmin_av": MSEmin_av,
    }


if __name__ == "__main__":
    main(seed=0, plot=True)
