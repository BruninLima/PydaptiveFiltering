# examples/example_channelEQ_CMA.py
#################################################################################
#                        Example: Channel Equalization                          #
#################################################################################
#                                                                               #
#  In this example we have a typical channel equalization scenario. We want     #
# to estimate the transmitted sequence with 4-QAM symbols. In                   #
# order to accomplish this task we use an adaptive filter with N coefficients.  #
# The procedure is:                                                             #
# 1)  Apply the originally transmitted signal distorted by the channel plus     #
#   environment noise as the input signal to an adaptive filter.                #
#   In this case, the transmitted signal is a random sequence with 4-QAM        #
#   symbols and unit variance. The channel is a multipath complex-valued        #
#   channel whose impulse response is h = [1.1+j*0.5, 0.1-j*0.3, -0.2-j*0.1]^T  #
#   In addition, the environment noise is AWGN with zero mean and               #
#   variance 10^(-2.5).                                                         #
# 2)  Choose an adaptive filtering algorithm to govern the rules of coefficient #
#   updating.                                                                   #
#                                                                               #
#     Adaptive Algorithm used here: CMA (Constant Modulus Algorithm)            #
#                                                                               #
#################################################################################

from __future__ import annotations

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.equalization import (
    qam4_constellation_unit_var,
    wiener_equalizer,
    generate_channel_data,
    simulate_constellations,
    ProgressConfig,
    report_progress,
)

def main(seed: int = 0, plot: bool = True):
    rng = np.random.default_rng(seed)

    ensemble, K, Ksim = 200, 10000, 400
    H = np.array([1.1 + 1j * 0.5, 0.1 - 1j * 0.3, -0.2 - 1j * 0.1], dtype=np.complex128)
    sigma_x2, sigma_n2 = 1.0, 10 ** (-2.5)
    N, mu, delay = 5, 0.001, 1

    constellation = qam4_constellation_unit_var()
    Wiener = wiener_equalizer(H, N, sigma_x2, sigma_n2, delay)

    W_final = np.zeros((N, ensemble), dtype=np.complex128)
    MSE = np.zeros((K - delay, ensemble), dtype=np.float64)

    cfg = ProgressConfig(verbose_progress=True, print_every=10, tail_window=500, optimize_verbose_first=True)

    t0 = perf_counter()
    for l in range(ensemble):
        t_real0 = perf_counter()

        _, x, _ = generate_channel_data(rng, K, H, sigma_n2, constellation)

        w_init = Wiener.ravel() + (rng.normal(0.0, 1.0, size=N) + 1j * rng.normal(0.0, 1.0, size=N)) / 4.0

        cma = pdf.CMA(
            filter_order=(N - 1),
            step_size=mu,
            w_init=w_init,
        )

        opt_verbose = bool(cfg.optimize_verbose_first and l == 0)
        res = cma.optimize(x[delay:], verbose=opt_verbose)

        e = np.asarray(res.errors).ravel()
        MSE[:, l] = (np.abs(e) ** 2)
        W_final[:, l] = cma.w

        report_progress(
            l=l,
            ensemble=ensemble,
            t0=t0,
            t_real0=t_real0,
            MSE_col=MSE[:, l],
            K_eff=K - delay,
            cfg=cfg,
        )

    total_time = perf_counter() - t0
    print(f"[Example/CMA] Total ensemble time: {total_time:.2f} s ({total_time/ensemble:.3f} s/realization)")

    W_av = np.mean(W_final, axis=1)
    MSE_av = np.mean(MSE, axis=1)

    equalizerInputMatrix, equalizerOutputVector, equalizerOutputVectorWiener = simulate_constellations(
        rng=rng,
        H=H,
        N=N,
        Ksim=Ksim,
        sigma_n2=sigma_n2,
        constellation=constellation,
        w_final=W_av,
        w_wiener=Wiener,
    )

    if plot:
        theta = np.linspace(-np.pi, np.pi, 200)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax1, ax2, ax3, ax4 = axes.ravel()

        ax1.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax1.scatter(np.real(equalizerOutputVector), np.imag(equalizerOutputVector), s=10)
        ax1.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax1.set_title("Equalizer output (CMA)")
        ax1.grid(True); ax1.set_aspect("equal", adjustable="box"); ax1.set_xlim([-2,2]); ax1.set_ylim([-2,2])

        ax2.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax2.scatter(np.real(equalizerInputMatrix.ravel()), np.imag(equalizerInputMatrix.ravel()), s=10)
        ax2.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax2.set_title("Equalizer input")
        ax2.grid(True); ax2.set_aspect("equal", adjustable="box"); ax2.set_xlim([-2,2]); ax2.set_ylim([-2,2])

        ax3.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax3.scatter(np.real(equalizerOutputVectorWiener), np.imag(equalizerOutputVectorWiener), s=10)
        ax3.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax3.set_title("Equalizer output (Wiener)")
        ax3.grid(True); ax3.set_aspect("equal", adjustable="box"); ax3.set_xlim([-2,2]); ax3.set_ylim([-2,2])

        ax4.semilogy(np.arange(1, K - delay + 1), np.abs(MSE_av))
        ax4.set_title("Learning curve (ensemble-averaged)")
        ax4.grid(True)

        fig.tight_layout()
        plt.show()

    return {"Wiener": Wiener, "W_av": W_av, "MSE_av": MSE_av, "W_final": W_final}


if __name__ == "__main__":
    main(seed=0, plot=True)
