# pydaptivefiltering/_utils/plotting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .metrics import db10, db20 
from .signal import freq_response_fir, freq_response_iir

__all__ = [
    "EqualizerDashboardConfig",
    "plot_equalizer_dashboard",
    "plot_system_id_single_figure",
    "plot_learning_curve",
]


@dataclass
class EqualizerDashboardConfig:
    """
    Configuration for a single-window equalizer dashboard.

    Parameters
    ----------
    n_freq : int
        Frequency grid size for frequency responses.
    zoom_start : int
        Zoom window start index (0-based, inclusive).
    zoom_end : int
        Zoom window end index (0-based, exclusive).
    show_stop_adaptation : bool
        If True, plots the ST accumulated error curves if provided.
    max_taps_to_show : int
        Show up to this many taps (w1, w2, w3, ...). Typically 3 like MATLAB.
    title : str
        Figure title.
    """
    n_freq: int = 1024
    zoom_start: int = 900
    zoom_end: int = 1000
    show_stop_adaptation: bool = True
    max_taps_to_show: int = 3
    title: str = "Kalman Equalizer vs RLS Equalizer"


def _clamp_zoom(zs: int, ze: int, T: int) -> Tuple[int, int]:
    zs = int(max(0, min(zs, max(T - 1, 0))))
    ze = int(max(zs + 1, min(ze, T)))
    return zs, ze


def plot_equalizer_dashboard(
    *,
    # core time-series
    d_full: np.ndarray,
    train_end: int,
    w_kal: np.ndarray,
    w_rls: np.ndarray,
    t_est_kal: np.ndarray,
    t_est_rls: np.ndarray,
    e_kal: np.ndarray,
    e_rls: np.ndarray,
    acc_err_kal: np.ndarray,
    acc_err_rls: np.ndarray,
    # stop-adaptation (optional)
    acc_err_kal_st: Optional[np.ndarray] = None,
    acc_err_rls_st: Optional[np.ndarray] = None,
    # channel response (for freq plot)
    num: Optional[np.ndarray] = None,
    den: Optional[np.ndarray] = None,
    # config + metadata
    cfg: Optional[EqualizerDashboardConfig] = None,
    meta_lines: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot a MATLAB-like equalizer comparison dashboard in a single figure.

    Expected shapes
    ---------------
    d_full, t_est_*, e_*, acc_err_*: (T,)
    w_kal, w_rls: (S, T)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    cfg = EqualizerDashboardConfig() if cfg is None else cfg

    d_full = np.asarray(d_full).ravel()
    t_est_kal = np.asarray(t_est_kal).ravel()
    t_est_rls = np.asarray(t_est_rls).ravel()
    e_kal = np.asarray(e_kal).ravel()
    e_rls = np.asarray(e_rls).ravel()
    acc_err_kal = np.asarray(acc_err_kal).ravel()
    acc_err_rls = np.asarray(acc_err_rls).ravel()

    w_kal = np.asarray(w_kal)
    w_rls = np.asarray(w_rls)

    if w_kal.ndim != 2 or w_rls.ndim != 2:
        raise ValueError("w_kal and w_rls must be 2D arrays (S,T).")

    S, T = w_kal.shape
    if w_rls.shape != (S, T):
        raise ValueError(f"w_rls shape {w_rls.shape} must match w_kal shape {(S, T)}.")

    # x-axis in MATLAB-like 1-based
    x = np.arange(T) + 1
    marker_x = int(train_end) + 1

    # error in dB
    e2_kal_db = db10(e_kal**2)
    e2_rls_db = db10(e_rls**2)

    # zoom window
    zs, ze = _clamp_zoom(cfg.zoom_start, cfg.zoom_end, T)

    # frequency responses
    have_channel = (num is not None) and (den is not None)
    if have_channel:
        w_f, H_kal = freq_response_fir(w_kal[:, -1], n_freq=cfg.n_freq)
        _,   H_rls = freq_response_fir(w_rls[:, -1], n_freq=cfg.n_freq)
        w_c, H_ch  = freq_response_iir(np.asarray(num), np.asarray(den), n_freq=cfg.n_freq)
        H_kal_db = db20(H_kal)
        H_rls_db = db20(H_rls)
        H_ch_db  = db20(H_ch)

        # Most diagnostic check: product equalizer*channel (magnitude)
        # Note: FIR and IIR are sampled on potentially identical grids; we assume same n_freq and range.
        # If grids differ, fall back to plotting only eq and channel.
        product_ok = (w_f.shape == w_c.shape) and np.allclose(w_f, w_c)
        if product_ok:
            H_tot_kal_db = db20(H_kal * H_ch)
            H_tot_rls_db = db20(H_rls * H_ch)
        else:
            H_tot_kal_db = None
            H_tot_rls_db = None
    else:
        w_f = H_kal_db = H_rls_db = H_ch_db = None
        H_tot_kal_db = H_tot_rls_db = None

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[1.0, 1.0, 1.1],
        width_ratios=[1.25, 1.0, 1.0, 1.0],
        hspace=0.35, wspace=0.30
    )

    ax_acc  = fig.add_subplot(gs[0, 0:2])
    ax_e2db = fig.add_subplot(gs[0, 2:4])

    # taps (up to 3 like MATLAB)
    ax_taps = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    ax_sym  = fig.add_subplot(gs[1, 3])

    ax_freq = fig.add_subplot(gs[2, 0:3])
    ax_meta = fig.add_subplot(gs[2, 3])

    # (1) accumulated error
    ax_acc.plot(x, acc_err_kal, label="Kalman", linewidth=2)
    ax_acc.plot(x, acc_err_rls, label="RLS", linestyle="--", linewidth=2)

    if cfg.show_stop_adaptation and (acc_err_kal_st is not None) and (acc_err_rls_st is not None):
        ax_acc.plot(x, np.asarray(acc_err_kal_st).ravel(), label="Kalman (ST)", linestyle=":", linewidth=2)
        ax_acc.plot(x, np.asarray(acc_err_rls_st).ravel(), label="RLS (ST)", linestyle="-.", linewidth=2)

    ax_acc.axvline(marker_x, linestyle="--", linewidth=2)
    ax_acc.set_title("Accumulated Error")
    ax_acc.set_xlabel("k")
    ax_acc.set_ylabel("acc error")
    ax_acc.grid(True)
    ax_acc.legend(loc="best")

    # (2) 10log10(|e|^2)
    ax_e2db.plot(x, e2_kal_db, label="Kalman", linewidth=2)
    ax_e2db.plot(x, e2_rls_db, label="RLS", linestyle="--", linewidth=2)
    ax_e2db.axvline(marker_x, linestyle="--", linewidth=2)
    ax_e2db.set_title(r"$10\log_{10}(|e(k)|^2)$")
    ax_e2db.set_xlabel("k")
    ax_e2db.grid(True)
    ax_e2db.legend(loc="best")

    # (3) taps w1,w2,w3
    for tap_i, ax in enumerate(ax_taps):
        if tap_i < min(S, cfg.max_taps_to_show):
            ax.plot(x, w_kal[tap_i, :], label="Kalman", linewidth=2)
            ax.plot(x, w_rls[tap_i, :], label="RLS", linestyle="--", linewidth=2)
            ax.axvline(marker_x, linestyle="--", linewidth=1.5)
            ax.set_title(f"w{tap_i+1}(k)")
            ax.set_xlabel("k")
            ax.grid(True)
            if tap_i == 0:
                ax.legend(loc="best")
        else:
            ax.axis("off")

    # (4) symbol decisions zoom
    # Keep this readable: show true as dots, and show mismatches for RLS/Kalman as markers.
    true_sym = np.sign(d_full[zs:ze])
    rls_sym = np.sign(t_est_rls[zs:ze])
    kal_sym = np.sign(t_est_kal[zs:ze])

    kk = x[zs:ze]
    ax_sym.plot(kk, true_sym, linestyle="None", marker="o", markersize=4, label="true")

    rls_err_idx = np.where(rls_sym != true_sym)[0]
    kal_err_idx = np.where(kal_sym != true_sym)[0]

    if rls_err_idx.size > 0:
        ax_sym.plot(kk[rls_err_idx], rls_sym[rls_err_idx], linestyle="None", marker="x", label="RLS err")
    if kal_err_idx.size > 0:
        ax_sym.plot(kk[kal_err_idx], kal_sym[kal_err_idx], linestyle="None", marker="s", label="Kalman err")

    ax_sym.axvline(marker_x, linestyle="--", linewidth=1.0)
    ax_sym.set_title(f"Symbol decisions (zoom {zs+1}:{ze})")
    ax_sym.set_xlabel("k")
    ax_sym.set_ylim(-1.5, 1.5)
    ax_sym.grid(True)
    ax_sym.legend(loc="best", fontsize=9)

    # (5) frequency response
    ax_freq.set_title("Frequency response")
    ax_freq.set_xlabel("Frequency (×π rad/sample)")
    ax_freq.set_ylabel("Magnitude (dB)")
    ax_freq.grid(True)

    if have_channel and (w_f is not None):
        ax_freq.plot(w_f / np.pi, H_kal_db, label="Kalman equalizer", linewidth=2)
        ax_freq.plot(w_f / np.pi, H_rls_db, label="RLS equalizer", linestyle="--", linewidth=2)
        ax_freq.plot(w_c / np.pi, H_ch_db, label="Channel", linestyle=":", linewidth=2)

        # bonus diagnostic: equalizer * channel should be flatter
        if (H_tot_kal_db is not None) and (H_tot_rls_db is not None):
            ax_freq.plot(w_f / np.pi, H_tot_kal_db, label="Kalman×Channel", linestyle="-.", linewidth=1.5)
            ax_freq.plot(w_f / np.pi, H_tot_rls_db, label="RLS×Channel", linestyle=(0, (1, 2)), linewidth=1.5)

        ax_freq.legend(loc="best")
    else:
        ax_freq.text(0.02, 0.95, "Channel (num/den) not provided.\nSkipping freq response.", va="top")

    # meta panel
    ax_meta.axis("off")
    if meta_lines is None:
        meta_lines = []
    meta_text = "\n".join([str(s) for s in meta_lines])
    ax_meta.text(0.02, 0.98, meta_text, va="top", ha="left", fontsize=10)

    fig.suptitle(cfg.title, fontsize=16)
    fig.tight_layout()
    plt.show()





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

def plot_learning_curve(MSE_av: np.ndarray, MSEmin_av: Optional[np.ndarray] = None, title: str = "") -> None:
    import matplotlib.pyplot as plt

    MSE_av = np.asarray(MSE_av, dtype=float).ravel()
    x = np.arange(1, MSE_av.size + 1)

    plt.figure()
    plt.semilogy(x, np.abs(MSE_av))
    if MSEmin_av is not None:
        MSEmin_av = np.asarray(MSEmin_av, dtype=float).ravel()
        plt.semilogy(x, np.abs(MSEmin_av))
        plt.legend(["MSE", "Noise floor"])
    plt.grid(True)
    plt.title(title or "Learning curve")
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.show()
