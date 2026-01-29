#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
#  to estimate the filter coefficients of an unknown system given by Wo.        #
#                                                                               #
#     Adaptive Algorithm used here: SimplifiedSMPUAP                                 #
#                                                                               #
#################################################################################

from __future__ import annotations

from time import perf_counter
import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    generate_qam4_input,
    build_desired_from_fir,
    pack_theta_from_result,
    )
from pydaptivefiltering._utils.progress import (    
    ProgressConfig,
    report_progress,
)
from pydaptivefiltering._utils.plotting import plot_system_id_single_figure

def main(seed: int = 0, plot: bool = True):
    rng_master = np.random.default_rng(seed)

    ensemble = 100
    K = 500
    sigma_n2 = 0.04
    N = 4

    Wo = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], dtype=np.complex128)

    W = np.zeros((N, K + 1, ensemble), dtype=np.complex128)
    MSE = np.zeros((K, ensemble), dtype=float)
    MSE_aux = np.zeros((K, ensemble), dtype=float)
    MSEmin = np.zeros((K, ensemble), dtype=float)

    cfg = ProgressConfig(verbose_progress=True, print_every=10, tail_window=50)

    t0 = perf_counter()

    for l in range(ensemble):
        t_real0 = perf_counter()
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        x = generate_qam4_input(rng, K)
        d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)

        up_selector = rng.integers(0, 2, size=(4, K), dtype=np.int8)
        up_selector[0, :] = 1
        flt = pdf.SimplifiedSMPUAP(gamma_bar=1.0, gamma=0.001, L=2, filter_order=3, up_selector=up_selector)
        res = flt.optimize(x.astype(np.complex128), d.astype(np.complex128), verbose=(l == 0))

        e = np.asarray(res.errors).ravel()
        MSE[:, l] = np.abs(e) ** 2
        MSE_aux[:, l] = MSE[:, l]
        MSEmin[:, l] = np.abs(n) ** 2

        W[:, :, l] = pack_theta_from_result(res=res, w_last=flt.w, n_coeffs=N, K=K)

        report_progress(
            algo_tag="SimplifiedSMPUAP",
            l=l, ensemble=ensemble, t0=t0, t_real0=t_real0,
            mse_col=MSE[:, l], cfg=cfg,
        )

    theta_av = np.mean(W, axis=2)
    MSE_av = np.mean(MSE, axis=1)
    MSE_aux_av = np.mean(MSE_aux, axis=1)
    MSEmin_av = np.mean(MSEmin, axis=1)

    print(f"[Example/SimplifiedSMPUAP] Total ensemble time: {perf_counter() - t0:.2f} s")

    if plot:
        plot_system_id_single_figure(
            MSE_av=MSE_av,
            MSEE_av=MSE_aux_av,
            MSEmin_av=MSEmin_av,
            theta_av=theta_av,
            poles_order=0,
            title_prefix="SimplifiedSMPUAP",
            show_complex_coeffs=True,
        )

    return {
        "Wo": Wo,
        "theta_av": theta_av,
        "MSE_av": MSE_av,
        "MSE_aux_av": MSE_aux_av,
        "MSEmin_av": MSEmin_av,
    }

if __name__ == "__main__":
    main(seed=0, plot=True)
