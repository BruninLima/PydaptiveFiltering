# examples/example_systemID_steiglitz_mcbride.py
#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
# to estimate coefficients related to an unknown system given by Wo.            #
# The procedure is:                                                             #
# 1)  Excitate both filters (the unknown and the adaptive) with the signal x.   #
#   Here, x is sign(randn) (MATLAB example), with variance ~1.                  #
# 2)  Generate desired signal, d = Wo' x + n, with noise power sigma_n2.        #
# 3)  Choose an adaptive filtering algorithm.                                   #
#                                                                               #
#     Adaptive Algorithm used here: SteiglitzMcBride                             #
#                                                                               #
#################################################################################

from __future__ import annotations

from time import perf_counter
import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.system_id import (
    generate_sign_input,
    build_desired_from_fir,
    pack_theta_from_result,
    ProgressConfig,
    report_progress,
    plot_system_id_single_figure,
)


def main(seed: int = 0, plot: bool = True):
    rng_master = np.random.default_rng(seed)

    # ----------------------------
    # Definitions (MATLAB-like)
    # ----------------------------
    ensemble = 100
    K = 500

    H = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], dtype=np.complex128)
    Wo = np.real(H).astype(float)  # unknown system (FIR, real)

    sigma_n2 = 0.04

    # Orders for adaptive IIR
    # MATLAB convention in your other examples:
    #   M = numerator order, N = denominator order
    M = 3
    N = 2

    # Steiglitz–McBride in your implementation:
    mu = 1e-3  # step_size

    if not hasattr(pdf, "SteiglitzMcBride"):
        raise RuntimeError("pydaptivefiltering does not expose 'SteiglitzMcBride'.")

    # ----------------------------
    # Memory allocation
    # ----------------------------
    n_coeffs = N + (M + 1)  # poles + (zeros+1)
    theta = np.zeros((n_coeffs, K + 1, ensemble), dtype=float)

    MSE = np.zeros((K, ensemble), dtype=float)
    MSEE = np.zeros((K, ensemble), dtype=float)   # we'll store |auxiliary_error|^2
    MSEmin = np.zeros((K, ensemble), dtype=float)

    cfg = ProgressConfig(verbose_progress=True, print_every=10, tail_window=50)

    # ----------------------------
    # Ensemble loop
    # ----------------------------
    t0 = perf_counter()

    for l in range(ensemble):
        t_real0 = perf_counter()

        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        # real-only signals
        x = generate_sign_input(rng, K)
        d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)

        sm = pdf.SteiglitzMcBride(
            zeros_order=M,
            poles_order=N,
            step_size=mu,
        )

        # we want auxiliary_error for MSEE curve
        res = sm.optimize(x, d, verbose=(l == 0), return_internal_states=True)

        e = np.asarray(res.errors).ravel()
        aux = np.asarray(res.extra.get("auxiliary_error", np.zeros_like(e))).ravel()

        MSE[:, l] = (np.abs(e) ** 2)
        MSEE[:, l] = (np.abs(aux) ** 2)
        MSEmin[:, l] = (np.abs(n) ** 2)

        theta[:, :, l] = pack_theta_from_result(
            res=res,
            w_last=sm.w,
            n_coeffs=n_coeffs,
            K=K,
        )

        report_progress(
            algo_tag="SteiglitzMcBride",
            l=l,
            ensemble=ensemble,
            t0=t0,
            t_real0=t_real0,
            mse_col=MSE[:, l],
            cfg=cfg,
        )

    total_time = perf_counter() - t0
    print(
        f"[Example/SteiglitzMcBride] Total ensemble time: {total_time:.2f} s "
        f"({total_time/ensemble:.3f} s/realization)"
    )

    # ----------------------------
    # Averaging (MATLAB)
    # ----------------------------
    theta_av = np.mean(theta, axis=2)  # (n_coeffs, K+1)
    MSE_av = np.mean(MSE, axis=1)
    MSEE_av = np.mean(MSEE, axis=1)
    MSEmin_av = np.mean(MSEmin, axis=1)

    # Print final coefficients like MATLAB
    print(f"\nAdaptive Filter Coefficients (last iteration computed over {ensemble} runs):")
    a_hat = theta_av[:N, -1]
    b_hat = theta_av[N:, -1]
    print("Numerator coefficients (direct part) b:")
    print(b_hat)
    print("Denominator coefficients (recursive part) a (stored as w[:N]):")
    print(a_hat)
    print("Unknown system (FIR) Wo:")
    print(Wo)

    # ----------------------------
    # Plot (single window)
    # ----------------------------
    if plot:
        plot_system_id_single_figure(
            MSE_av=MSE_av,
            MSEE_av=MSEE_av,       # auxiliary/filtered equation error curve
            MSEmin_av=MSEmin_av,
            theta_av=theta_av,
            poles_order=N,
            title_prefix="SteiglitzMcBride",
        )

    return {
        "Wo": Wo,
        "theta_av": theta_av,
        "MSE_av": MSE_av,
        "MSEE_av": MSEE_av,
        "MSEmin_av": MSEmin_av,
    }


if __name__ == "__main__":
    main(seed=0, plot=True)
