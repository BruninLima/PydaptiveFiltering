# examples/example_equalizer_kalman_vs_rls.py
#################################################################################
#                      Example: Kalman Equalizer vs RLS Equalizer               #
#################################################################################
#                                                                               #
#  Inspired by a MATLAB example comparing a Kalman-based FIR equalizer and an   #
#  RLS FIR equalizer for a channel with ISI.                                    #
#                                                                               #
#  Setup:                                                                       #
#   - Training: first half uses known training symbols t[k]                     #
#   - Test: second half uses decision-directed updates (sign of estimate)       #
#                                                                               #
#################################################################################

from __future__ import annotations

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import pydaptivefiltering as pdf


def sign01(x: np.ndarray) -> np.ndarray:
    """MATLAB-like sign for real: returns +/-1 (maps 0 -> 1)."""
    y = np.sign(x).astype(float)
    y[y == 0.0] = 1.0
    return y


def tapped_delay_matrix(x: np.ndarray, S: int) -> np.ndarray:
    """
    Create tapped delay line matrix like MATLAB buffer(x,S,S-1,'nodelay') but as (T, S).
    Row k corresponds to [x[k+S-1], x[k+S-2], ..., x[k]]  (most recent first).
    """
    x = np.asarray(x).ravel()
    N = x.size
    T = N - S + 1
    X = np.zeros((T, S), dtype=x.dtype)
    for k in range(T):
        X[k, :] = x[k : k + S][::-1]
    return X


def freq_response_fir(b: np.ndarray, n_freq: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    FIR frequency response H(e^jw) on w in [0, pi].
    b is assumed in the usual FIR form:
        y[n] = sum_{m=0}^{M} b[m] x[n-m]
    Here, our equalizer weights are for regressor [x[n], x[n-1], ...],
    so b[m] corresponds to tap for delay m (which matches this convention).
    """
    b = np.asarray(b).ravel().astype(complex)
    w = np.linspace(0.0, np.pi, n_freq)
    m = np.arange(b.size)
    E = np.exp(-1j * np.outer(w, m))
    H = E @ b
    return w, H


def freq_response_iir(num: np.ndarray, den: np.ndarray, n_freq: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    IIR frequency response H(e^jw) = B(e^jw)/A(e^jw) on w in [0, pi],
    using direct polynomial evaluation (no scipy).
    """
    num = np.asarray(num).ravel().astype(complex)
    den = np.asarray(den).ravel().astype(complex)

    w = np.linspace(0.0, np.pi, n_freq)
    k_num = np.arange(num.size)
    k_den = np.arange(den.size)

    E_num = np.exp(-1j * np.outer(w, k_num))
    E_den = np.exp(-1j * np.outer(w, k_den))

    B = E_num @ num
    A = E_den @ den

    # guard against tiny denominators
    A = np.where(np.abs(A) < 1e-15, A + 1e-15, A)
    H = B / A
    return w, H


def _fill_w_history(w_hist: np.ndarray, coeffs: np.ndarray, n_steps: int) -> None:
    """
    Fill w_hist[:, :n_steps] from coeffs with robust shape handling.
    Expected coeffs shapes (any one):
      - (n_steps, S)  : per-iter rows
      - (S, n_steps)  : per-iter cols
      - (n_steps+1,S) : includes initial w(0) then w(1)... => drop first
      - (S,n_steps+1) : includes initial then ... => drop first
    """
    S, _T = w_hist.shape
    C = np.asarray(coeffs)
    if C.ndim != 2:
        return

    if C.shape[0] == n_steps + 1 and C.shape[1] == S:
        C = C[1:, :]
    elif C.shape[1] == n_steps + 1 and C.shape[0] == S:
        C = C[:, 1:]

    if C.shape[0] == n_steps and C.shape[1] == S:
        w_hist[:, :n_steps] = C.T.real
    elif C.shape[0] == S and C.shape[1] == n_steps:
        w_hist[:, :n_steps] = C.real
    else:
        # fallback: replicate last vector if possible
        if C.shape[1] == S:
            w_hist[:, :n_steps] = np.tile(C[-1, :].real.reshape(-1, 1), (1, n_steps))
        elif C.shape[0] == S:
            w_hist[:, :n_steps] = np.tile(C[:, -1].real.reshape(-1, 1), (1, n_steps))


def main(
    seed: int = 0,
    plot: bool = True,
    N_samples: int = 1000,
    N: int = 2,             # filter order -> S = N+1 taps
    L: int = 1,             # decision delay
    sigma_n2: float = 0.2,  # measurement noise variance (R_n)
    lambda_: float = 0.99,  # forgetting factor for RLS
    delta_rls: float = 1.0, # MATLAB uses S_d = I, which corresponds to delta=1.0
    q_process: float = 0.0, # MATLAB script effectively uses Q=0 in the Kalman covariance recursion
    show_stop_adaptation: bool = True,  # plot ST curves too (optional)
    n_freq: int = 1024,
    zoom_start: int = 900,  # similar to MATLAB's [900 1000] window
    zoom_end: int = 1000,
):
    rng = np.random.default_rng(seed)

    # ----------------------------
    # Signals and channel
    # ----------------------------
    S = N + 1  # number of taps

    # training symbols (+/-1)
    t = sign01(rng.standard_normal(N_samples))

    # channel (IIR) like MATLAB:
    num = np.array([0.0, 1.6561], dtype=float)
    den = np.array([1.0, 0.0, 0.81], dtype=float)

    # r_aux = filter(num,den,t) without scipy
    r_aux = np.zeros_like(t)
    for k in range(N_samples):
        xk = t[k]
        xk_1 = t[k - 1] if k - 1 >= 0 else 0.0
        yk_1 = r_aux[k - 1] if k - 1 >= 0 else 0.0
        yk_2 = r_aux[k - 2] if k - 2 >= 0 else 0.0
        r_aux[k] = num[0] * xk + num[1] * xk_1 - den[1] * yk_1 - den[2] * yk_2

    sigma_n = float(np.sqrt(sigma_n2))
    r = r_aux + sigma_n * rng.standard_normal(N_samples)

    # tapped delay lines: (T,S)
    R_tdl = tapped_delay_matrix(r, S)
    T = R_tdl.shape[0]

    # desired aligned with delay L: MATLAB uses t(k+L)
    if T + L > len(t):
        raise ValueError("Need N_samples large enough so that (N_samples - S + 1 + L) <= N_samples.")
    d_full = t[L : L + T].copy()  # (T,)

    # train/test split like MATLAB:
    train_end = (N_samples // 2) - S + 1
    train_end = int(max(0, min(train_end, T)))  # number of training iterations
    test_start = train_end  # 0-based index where DD starts

    # ----------------------------
    # Kalman equalizer (MATLAB-like)
    # ----------------------------
    # State covariance recursion:
    #   MATLAB code does: R_e <- (I-KC^T) R_e
    # with K = R_e C / (C^T R_e C + R_n), i.e. no +Q term.
    # We keep q_process parameter for optional numerical stability, but default is 0.0.
    R_n = float(sigma_n2)
    R_e = np.eye(S, dtype=float)  # covariance
    w_kal = np.zeros((S, T), dtype=float)
    w_prev_kal = np.zeros(S, dtype=float)

    # ----------------------------
    # RLS equalizer via pdf.RLS + manual DD continuation
    # ----------------------------
    # Align scalar stream for FIR memory: x_rls[k] = r[k+S-1]
    x_rls = r[S - 1 : S - 1 + T]  # shape (T,)

    rls = pdf.RLS(
        filter_order=N,
        delta=delta_rls,
        forgetting_factor=lambda_,
    )

    w_rls = np.zeros((S, T), dtype=float)

    # "Stop adaptation" copies (MATLAB's *_st)
    w_kal_st = np.zeros((S, T), dtype=float)
    w_rls_st = np.zeros((S, T), dtype=float)

    # ----------------------------
    # Bookkeeping arrays (like MATLAB)
    # ----------------------------
    t_est_kal = np.zeros(T, dtype=float)
    t_est_rls = np.zeros(T, dtype=float)
    e_kal = np.zeros(T, dtype=float)
    e_rls = np.zeros(T, dtype=float)

    # Stop-adaptation estimates
    t_est_kal_st = np.zeros(T, dtype=float)
    t_est_rls_st = np.zeros(T, dtype=float)
    e_kal_st = np.zeros(T, dtype=float)
    e_rls_st = np.zeros(T, dtype=float)

    n_err_kal = np.zeros(T, dtype=float)
    n_err_rls = np.zeros(T, dtype=float)
    n_err_kal_st = np.zeros(T, dtype=float)
    n_err_rls_st = np.zeros(T, dtype=float)

    # ----------------------------
    # RUN
    # ----------------------------
    t0 = perf_counter()

    # --- TRAIN RLS using library optimize (known desired)
    if train_end > 0:
        res_rls_train = rls.optimize(
            x_rls[:train_end],
            d_full[:train_end],
            verbose=True,
            return_internal_states=True,
        )
        if getattr(res_rls_train, "coefficients", None) is not None:
            _fill_w_history(w_rls, np.asarray(res_rls_train.coefficients), train_end)
        else:
            w_rls[:, :train_end] = np.tile(np.asarray(rls.w).real.reshape(-1, 1), (1, train_end))

    # Initialize stop-adaptation taps at end of training
    if train_end > 0:
        w_kal_st[:, :train_end] = w_kal[:, :train_end]  # will fill kalman during loop below
        w_rls_st[:, :train_end] = w_rls[:, :train_end]

    # --- TRAIN loop for Kalman (and for computing t_est/e/n_err for both)
    # We compute Kalman in the loop (to match MATLAB step-by-step).
    for k in range(train_end):
        C = R_tdl[k, :].reshape(-1, 1)  # (S,1)
        d_true = float(d_full[k])

        # ===== Kalman (training uses true desired) =====
        # Optional +Q (process noise) to match or stabilize; MATLAB is Q=0
        if q_process != 0.0:
            R_e = R_e + float(q_process) * np.eye(S)

        denom = float(C.T @ R_e @ C + R_n)
        K = (R_e @ C) / denom  # (S,1)
        w_new = w_prev_kal + (K[:, 0] * (d_true - float(C[:, 0] @ w_prev_kal)))
        R_e = (np.eye(S) - (K @ C.T)) @ R_e

        w_kal[:, k] = w_new
        w_prev_kal = w_new

        # Kalman output estimate
        t_est_kal[k] = float(C[:, 0] @ w_new)
        e_kal[k] = d_true - t_est_kal[k]
        d_hat_kal = float(np.sign(t_est_kal[k]) if t_est_kal[k] != 0 else 1.0)
        n_err_kal[k] = 1.0 if (d_true != d_hat_kal) else 0.0

        # ===== RLS metrics during training (weights already from pdf.RLS history) =====
        t_est_rls[k] = float(R_tdl[k, :] @ w_rls[:, k])
        e_rls[k] = d_true - t_est_rls[k]
        d_hat_rls = float(np.sign(t_est_rls[k]) if t_est_rls[k] != 0 else 1.0)
        n_err_rls[k] = 1.0 if (d_true != d_hat_rls) else 0.0

        # stop-adaptation (same as adaptive during training)
        w_kal_st[:, k] = w_kal[:, k]
        w_rls_st[:, k] = w_rls[:, k]
        t_est_kal_st[k] = t_est_kal[k]
        t_est_rls_st[k] = t_est_rls[k]
        e_kal_st[k] = e_kal[k]
        e_rls_st[k] = e_rls[k]
        n_err_kal_st[k] = n_err_kal[k]
        n_err_rls_st[k] = n_err_rls[k]

    # --- TEST (DD) for both + stop-adaptation curves
    for k in range(test_start, T):
        C = R_tdl[k, :].reshape(-1, 1)  # (S,1)
        d_true = float(d_full[k])

        # ===== Kalman DD =====
        if q_process != 0.0:
            R_e = R_e + float(q_process) * np.eye(S)

        y_hat = float(C[:, 0] @ w_prev_kal)
        d_dd = float(np.sign(y_hat) if y_hat != 0 else 1.0)

        denom = float(C.T @ R_e @ C + R_n)
        K = (R_e @ C) / denom
        w_new = w_prev_kal + (K[:, 0] * (d_dd - float(C[:, 0] @ w_prev_kal)))
        R_e = (np.eye(S) - (K @ C.T)) @ R_e

        w_kal[:, k] = w_new
        w_prev_kal = w_new

        t_est_kal[k] = float(C[:, 0] @ w_new)
        e_kal[k] = d_true - t_est_kal[k]
        d_hat_kal = float(np.sign(t_est_kal[k]) if t_est_kal[k] != 0 else 1.0)
        n_err_kal[k] = 1.0 if (d_true != d_hat_kal) else 0.0

        # ===== RLS DD (continue from trained rls state) =====
        # update FIR memory
        rls.regressor = np.roll(rls.regressor, 1)
        rls.regressor[0] = complex(x_rls[k])

        y_hat_c = complex(np.vdot(rls.w, rls.regressor))  # w^H x
        y_hat_r = float(y_hat_c.real)
        d_dd_rls = float(np.sign(y_hat_r) if y_hat_r != 0 else 1.0)

        e = complex(d_dd_rls) - y_hat_c

        Sx = rls.S_d @ rls.regressor
        den_rls = rls.forgetting_factor + complex(np.vdot(rls.regressor, Sx))
        if abs(den_rls) < 1e-12:
            den_rls = den_rls + (1e-12 + 0.0j)

        g = Sx / den_rls
        rls.w = rls.w + np.conj(e) * g
        rls.S_d = (rls.S_d - np.outer(g, np.conj(Sx))) / rls.forgetting_factor

        w_rls[:, k] = np.asarray(rls.w).real

        t_est_rls[k] = float(R_tdl[k, :] @ w_rls[:, k])
        e_rls[k] = d_true - t_est_rls[k]
        d_hat_rls = float(np.sign(t_est_rls[k]) if t_est_rls[k] != 0 else 1.0)
        n_err_rls[k] = 1.0 if (d_true != d_hat_rls) else 0.0

        # ===== Stop-adaptation (ST): keep last trained weights fixed =====
        if k == 0:
            # shouldn't happen because train_end >= 0, but guard anyway
            w_kal_st[:, k] = w_kal[:, k]
            w_rls_st[:, k] = w_rls[:, k]
        else:
            w_kal_st[:, k] = w_kal_st[:, k - 1]
            w_rls_st[:, k] = w_rls_st[:, k - 1]

        t_est_kal_st[k] = float(R_tdl[k, :] @ w_kal_st[:, k])
        e_kal_st[k] = d_true - t_est_kal_st[k]
        d_hat_kal_st = float(np.sign(t_est_kal_st[k]) if t_est_kal_st[k] != 0 else 1.0)
        n_err_kal_st[k] = 1.0 if (d_true != d_hat_kal_st) else 0.0

        t_est_rls_st[k] = float(R_tdl[k, :] @ w_rls_st[:, k])
        e_rls_st[k] = d_true - t_est_rls_st[k]
        d_hat_rls_st = float(np.sign(t_est_rls_st[k]) if t_est_rls_st[k] != 0 else 1.0)
        n_err_rls_st[k] = 1.0 if (d_true != d_hat_rls_st) else 0.0

    runtime = perf_counter() - t0
    print(f"[Example/Kalman vs RLS] Completed in {runtime:.3f} s")

    # accumulated errors
    acc_err_kal = np.cumsum(n_err_kal)
    acc_err_rls = np.cumsum(n_err_rls)
    acc_err_kal_st = np.cumsum(n_err_kal_st)
    acc_err_rls_st = np.cumsum(n_err_rls_st)

    # ----------------------------
    # PLOTS
    # ----------------------------
    if plot:
        import matplotlib.gridspec as gridspec

        # ---------- preparar curvas auxiliares ----------
        # erro em dB
        e2_kal_db = 10.0 * np.log10(e_kal**2 + 1e-20)
        e2_rls_db = 10.0 * np.log10(e_rls**2 + 1e-20)

        # freq response
        wf_kal = w_kal[:, -1]
        wf_rls = w_rls[:, -1]
        w_f, H_kal = freq_response_fir(wf_kal, n_freq=n_freq)
        _,   H_rls = freq_response_fir(wf_rls, n_freq=n_freq)
        w_c, H_ch  = freq_response_iir(num, den, n_freq=n_freq)

        H_kal_db = 20 * np.log10(np.abs(H_kal) + 1e-12)
        H_rls_db = 20 * np.log10(np.abs(H_rls) + 1e-12)
        H_ch_db  = 20 * np.log10(np.abs(H_ch)  + 1e-12)

        # zoom janela (clamp)
        zs = max(0, min(int(zoom_start), T - 1))
        ze = max(zs + 1, min(int(zoom_end), T))

        # ---------- figura única ----------
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(
            3, 4, figure=fig,
            height_ratios=[1.0, 1.0, 1.1],
            width_ratios=[1.25, 1.0, 1.0, 1.0],
            hspace=0.35, wspace=0.30
        )

        ax_acc   = fig.add_subplot(gs[0, 0:2])  # erro acumulado (topo-esq largo)
        ax_e2db  = fig.add_subplot(gs[0, 2:4])  # 10log|e|^2 (topo-dir largo)

        ax_w0    = fig.add_subplot(gs[1, 0])    # w1
        ax_w1    = fig.add_subplot(gs[1, 1])    # w2
        ax_w2    = fig.add_subplot(gs[1, 2])    # w3
        ax_sym   = fig.add_subplot(gs[1, 3])    # zoom símbolos

        ax_freq  = fig.add_subplot(gs[2, 0:3])  # freq response (bottom largo)
        ax_dummy = fig.add_subplot(gs[2, 3])    # opcional: ST ou painel extra

        # ---------- (1) accumulated error ----------
        x = np.arange(T) + 1
        ax_acc.plot(x, acc_err_kal, label="Kalman", linewidth=2)
        ax_acc.plot(x, acc_err_rls, label="RLS", linestyle="--", linewidth=2)
        if show_stop_adaptation:
            ax_acc.plot(x, acc_err_kal_st, label="Kalman (ST)", linestyle=":", linewidth=2)
            ax_acc.plot(x, acc_err_rls_st, label="RLS (ST)", linestyle="-.", linewidth=2)

        o_1based = train_end + 1
        ax_acc.axvline(o_1based, linestyle="--", linewidth=2)
        ax_acc.set_title("Accumulated Error")
        ax_acc.set_xlabel("k")
        ax_acc.set_ylabel("acc error")
        ax_acc.grid(True)
        ax_acc.legend(loc="best")

        # ---------- (2) 10*log10(|e|^2) ----------
        ax_e2db.plot(x, e2_kal_db, label="Kalman", linewidth=2)
        ax_e2db.plot(x, e2_rls_db, label="RLS", linestyle="--", linewidth=2)
        ax_e2db.axvline(o_1based, linestyle="--", linewidth=2)
        ax_e2db.set_title(r"$10\log_{10}(|e(k)|^2)$")
        ax_e2db.set_xlabel("k")
        ax_e2db.grid(True)
        ax_e2db.legend(loc="best")

        # ---------- (3) taps ----------
        taps_axes = [ax_w0, ax_w1, ax_w2]
        for tap_i, ax in enumerate(taps_axes):
            if tap_i < S:
                ax.plot(x, w_kal[tap_i, :], label="Kalman", linewidth=2)
                ax.plot(x, w_rls[tap_i, :], label="RLS", linestyle="--", linewidth=2)
                ax.axvline(o_1based, linestyle="--", linewidth=1.5)
                ax.set_title(f"w{tap_i+1}(k)")
                ax.set_xlabel("k")
                ax.grid(True)
                if tap_i == 0:
                    ax.legend(loc="best")
            else:
                ax.axis("off")

        # ---------- (4) símbolos (zoom) ----------
        ax_sym.plot(
            x[zs:ze],
            np.sign(d_full[zs:ze]),
            marker="+",
            linestyle="-",
            linewidth=2,
            label="true",
        )
        ax_sym.plot(
            x[zs:ze],
            np.sign(t_est_rls[zs:ze]),
            marker="x",
            linestyle="--",
            linewidth=2,
            label="RLS",
        )
        ax_sym.plot(
            x[zs:ze],
            np.sign(t_est_kal[zs:ze]),
            marker="o",
            linestyle=":",
            linewidth=2,
            label="Kalman",
        )
        ax_sym.set_title(f"Symbol decisions (zoom {zs+1}:{ze})")
        ax_sym.set_xlabel("k")
        ax_sym.set_ylim(-1.5, 1.5)
        ax_sym.grid(True)
        ax_sym.legend(loc="best", fontsize=9)

        # ---------- (5) freq response ----------
        ax_freq.plot(w_f / np.pi, H_kal_db, label="Kalman equalizer", linewidth=2)
        ax_freq.plot(w_f / np.pi, H_rls_db, label="RLS equalizer", linestyle="--", linewidth=2)
        ax_freq.plot(w_c / np.pi, H_ch_db, label="Channel", linestyle=":", linewidth=2)
        ax_freq.set_title("Frequency response")
        ax_freq.set_xlabel("Frequency (×π rad/sample)")
        ax_freq.set_ylabel("Magnitude (dB)")
        ax_freq.grid(True)
        ax_freq.legend(loc="best")

        # ---------- (6) painel extra (ST zoom ou vazio) ----------
        ax_dummy.axis("off")
        ax_dummy.text(
            0.02, 0.98,
            f"N_samples={N_samples}\nN={N} (S={S})\nL={L}\nλ={lambda_}\nσ²={sigma_n2}\ntrain_end={train_end}",
            va="top", ha="left", fontsize=10
        )

        fig.suptitle("Kalman Equalizer vs RLS Equalizer", fontsize=16)
        fig.tight_layout()
        plt.show()


    return {
        "t": t,
        "r": r,
        "R_tdl": R_tdl,
        "d_full": d_full,
        "train_end": train_end,
        "w_kal": w_kal,
        "w_rls": w_rls,
        "w_kal_st": w_kal_st,
        "w_rls_st": w_rls_st,
        "t_est_kal": t_est_kal,
        "t_est_rls": t_est_rls,
        "e_kal": e_kal,
        "e_rls": e_rls,
        "acc_err_kal": acc_err_kal,
        "acc_err_rls": acc_err_rls,
        "acc_err_kal_st": acc_err_kal_st,
        "acc_err_rls_st": acc_err_rls_st,
    }


if __name__ == "__main__":
    main(seed=0, plot=True)
