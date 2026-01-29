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
    Create tapped delay line matrix like MATLAB buffer(x,S,S-1,'nodelay') but as (N-S+1, S).
    Row k corresponds to [x[k+S-1], x[k+S-2], ..., x[k]]^T  (most recent first).
    """
    x = np.asarray(x).ravel()
    N = x.size
    T = N - S + 1
    X = np.zeros((T, S), dtype=x.dtype)
    for k in range(T):
        X[k, :] = x[k : k + S][::-1]
    return X


def main(
    seed: int = 0,
    plot: bool = True,
    N_samples: int = 1000,
    N: int = 2,            # filter order, so S = N+1 taps
    L: int = 1,            # decision delay
    sigma_n2: float = 0.2, # MATLAB example uses sigma_n=sqrt(0.2)
    lambda_: float = 0.99, # forgetting factor for RLS
):
    rng = np.random.default_rng(seed)

    # ----------------------------
    # Signals and channel
    # ----------------------------
    S = N + 1  # number of taps

    # training symbols (+/-1)
    t = sign01(rng.standard_normal(N_samples))

    # channel (IIR) like MATLAB:
    # num = [0 1.6561], den = [1 0 0.81]
    num = np.array([0.0, 1.6561], dtype=float)
    den = np.array([1.0, 0.0, 0.81], dtype=float)

    # generate channel output r_aux = filter(num, den, t)
    # We'll implement the IIR recursion directly to avoid scipy dependency.
    # y[k] = num[0]*x[k] + num[1]*x[k-1] - den[1]*y[k-1] - den[2]*y[k-2]
    r_aux = np.zeros_like(t)
    for k in range(N_samples):
        xk = t[k]
        xk_1 = t[k - 1] if k - 1 >= 0 else 0.0
        yk_1 = r_aux[k - 1] if k - 1 >= 0 else 0.0
        yk_2 = r_aux[k - 2] if k - 2 >= 0 else 0.0
        r_aux[k] = num[0] * xk + num[1] * xk_1 - den[1] * yk_1 - den[2] * yk_2

    # add noise
    sigma_n = float(np.sqrt(sigma_n2))
    noise = sigma_n * rng.standard_normal(N_samples)
    r = r_aux + noise

    # tapped delay lines: shape (T, S) where T = N_samples - S + 1
    R_tdl = tapped_delay_matrix(r, S)
    T = R_tdl.shape[0]

    # desired sequence aligned with delay L:
    # MATLAB uses t(k+L) where k indexes columns of r_tdl.
    # Here k goes from 0..T-1, so desired is t[k+L].
    # Need t length >= T + L
    if T + L > len(t):
        raise ValueError("Need N_samples large enough so that (N_samples - S + 1 + L) <= N_samples.")

    d_full = t[L : L + T].copy()  # (T,)

    # split training / test like MATLAB:
    train_end = (N_samples // 2) - S + 1  # MATLAB loop end index (1-based)
    train_end = int(max(0, min(train_end, T)))  # clamp to [0, T]
    # training indices: [0, train_end)
    # test indices: [train_end, T)

    # ----------------------------
    # RLS equalizer (FIR)
    # ----------------------------
    # If your library exposes an RLS FIR class, use it.
    # Otherwise, we'll implement minimal RLS here (same as MATLAB).
    w_rls = np.zeros((S, T), dtype=float)
    Sd = np.eye(S, dtype=float)

    # ----------------------------
    # Kalman equalizer as "state estimator"
    # ----------------------------
    # We model the equalizer taps as a random-walk state:
    #   w(k) = w(k-1) + q_noise,  A = I, B = I, Rn = Q
    # measurement:
    #   d(k) = C^T(k) w(k) + v(k) ,  C^T(k) = r_tdl(:,k)^T
    #
    # So:
    #   A: (S,S) identity
    #   C_T: (1,S) per k
    #   Rn: (S,S) small process noise covariance
    #   Rn1: (1,1) measurement noise variance
    #
    A = np.eye(S, dtype=float)

    # choose a small Q (process noise) -> controls tracking (taps drift)
    # MATLAB didn't specify Q (implicitly 0). Using a tiny value helps numerical stability.
    q = 1e-6
    Rn = (q * np.eye(S, dtype=float))

    Rn1 = np.array([[sigma_n2]], dtype=float)  # measurement noise variance

    # Build time-varying C_T list: each is (1,S)
    C_T_seq = [R_tdl[k, :].reshape(1, -1) for k in range(T)]

    kf = pdf.Kalman(
        A=A,
        C_T=C_T_seq,
        Rn=Rn,
        Rn1=Rn1,
        B=None,
        x_init=np.zeros(S, dtype=float),
        Re_init=np.eye(S, dtype=float),
    )

    # We'll run Kalman step-by-step by calling optimize once on the whole sequence,
    # but we need the desired sequence as measurements y[k].
    # Here, Kalman expects "input_signal" to be y[k] sequence (N,p). We'll pass d_full as (T,).
    # It will output state estimates w_hat(k|k) in outputs.

    # ----------------------------
    # Run: Training + Test (decision-directed)
    # ----------------------------
    # For a fair comparison with MATLAB, we do BOTH algorithms online with DD in test.
    # The Kalman class runs on whatever "measurement" y we feed it.
    #
    # We'll create y_meas sequence that is:
    #   training: y = d_full (true symbols)
    #   test:     y = sign(estimate) (decision-directed)
    #
    # But decision-directed needs estimates, so we iterate manually for both.
    #
    # Since your Kalman implementation currently runs a whole sequence inside optimize(),
    # we'll implement a tiny per-step wrapper here by reusing its internal matrices logic
    # would require changing the class.
    #
    # So: simplest is implement per-step Kalman update in the example (matching MATLAB),
    # and keep pdf.Kalman example separately.
    #
    # HOWEVER: you explicitly asked "um para o kalman" usando a sua classe.
    # Então aqui vai a versão que usa pdf.Kalman em 2 passes:
    #   - Pass 1 (training): run Kalman on y = true training symbols
    #   - Pass 2 (test): freeze last state OR run DD with a loop (manual) – aqui vou fazer
    #     DD manual para ficar fiel ao MATLAB.
    #
    # Resultado: ainda usamos pdf.Kalman para o trecho training (e para shapes e cov),
    # e fazemos DD manual no teste reaproveitando as mesmas equações (curto e claro).

    t0 = perf_counter()

    # --- RLS + Kalman TRAIN ---
    n_err_rls = np.zeros(T, dtype=float)
    n_err_kal = np.zeros(T, dtype=float)

    # Kalman: run training with true symbols
    res_train = kf.optimize(d_full[:train_end], verbose=True, return_internal_states=True)

    # store training states for Kalman
    w_kal = np.zeros((S, T), dtype=float)
    if train_end > 0:
        w_kal[:, :train_end] = np.asarray(res_train.outputs).T  # outputs: (train_end, S)

    # RLS training loop
    for k in range(train_end):
        C = R_tdl[k, :].reshape(-1, 1)  # (S,1)
        d_k = float(d_full[k])

        w_prev = w_rls[:, k - 1] if k > 0 else np.zeros(S, dtype=float)

        # priori error
        e_prio = d_k - float(w_prev @ C[:, 0])

        phi = Sd @ C
        den = float(lambda_ + (C.T @ phi))
        Sd = (1.0 / lambda_) * (Sd - (phi @ phi.T) / den)

        w_new = w_prev + (e_prio * (Sd @ C))[:, 0]
        w_rls[:, k] = w_new

        t_est = float(w_new @ C[:, 0])
        n_err_rls[k] = 1.0 if (d_k != float(np.sign(t_est) if t_est != 0 else 1.0)) else 0.0

    # --- Kalman TEST (decision-directed) ---
    # We will continue from the last Kalman training state:
    # Take last state/cov from kf (already updated by optimize)
    # and do per-step update with y = sign(C^T w_prev).

    # Recover last Re from coefficients history (stored as list of Re)
    # Kalman packs "coefficients" as ndarray in result; in your base, it's whatever _pack_results does.
    # We'll use internal kf.Re and kf.x (already final from training).
    #
    # For safety, if train_end == 0, kf is at init.

    R_e = np.asarray(kf.Re, dtype=float)
    w_prev_kal = np.asarray(kf.x[:, 0], dtype=float)

    for k in range(train_end, T):
        C = R_tdl[k, :].reshape(-1, 1)  # (S,1)

        # decision-directed "measurement"
        y_hat = float(C[:, 0] @ w_prev_kal)
        d_dd = float(np.sign(y_hat) if y_hat != 0 else 1.0)

        # Kalman gain for scalar measurement:
        # K = R_e C / (C^T R_e C + R_n)
        denom = float(C.T @ R_e @ C + sigma_n2)
        K_gain = (R_e @ C) / denom  # (S,1)

        w_new = w_prev_kal + (K_gain[:, 0] * (d_dd - float(C[:, 0] @ w_prev_kal)))
        R_e = (np.eye(S) - (K_gain @ C.T)) @ R_e

        w_kal[:, k] = w_new
        w_prev_kal = w_new

        # error count vs true desired (still d_full uses true symbols)
        d_true = float(d_full[k])
        n_err_kal[k] = 1.0 if (d_true != float(np.sign(float(C[:, 0] @ w_new)) if float(C[:, 0] @ w_new) != 0 else 1.0)) else 0.0

        # RLS decision-directed test (match MATLAB)
        w_prev = w_rls[:, k - 1] if k > 0 else np.zeros(S, dtype=float)

        y_hat_rls = float(w_prev @ C[:, 0])
        d_dd_rls = float(np.sign(y_hat_rls) if y_hat_rls != 0 else 1.0)

        e_prio = d_dd_rls - float(w_prev @ C[:, 0])

        phi = Sd @ C
        den = float(lambda_ + (C.T @ phi))
        Sd = (1.0 / lambda_) * (Sd - (phi @ phi.T) / den)

        w_new_rls = w_prev + (e_prio * (Sd @ C))[:, 0]
        w_rls[:, k] = w_new_rls

        d_true = float(d_full[k])
        t_est = float(w_new_rls @ C[:, 0])
        n_err_rls[k] = 1.0 if (d_true != float(np.sign(t_est) if t_est != 0 else 1.0)) else 0.0

    runtime = perf_counter() - t0
    print(f"[Example/Kalman vs RLS] Completed in {runtime:.3f} s")

    # accumulated errors
    acc_err_kal = np.cumsum(n_err_kal)
    acc_err_rls = np.cumsum(n_err_rls)

    # ----------------------------
    # Plots (single window-ish)
    # ----------------------------
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # (1) Accumulated error with train/test marker
        ax1.plot(np.arange(T), acc_err_kal, label="Kalman")
        ax1.plot(np.arange(T), acc_err_rls, label="RLS", linestyle="--")
        ax1.axvline(train_end, linestyle="--")
        ax1.set_title("Accumulated Error")
        ax1.set_xlabel("iteration k")
        ax1.set_ylabel("acc error")
        ax1.grid(True)
        ax1.legend()

        # (2) Instantaneous squared error (in dB)
        # innovation for Kalman: e = d_true - y_hat; for RLS: same
        e2_kal = np.zeros(T, dtype=float)
        e2_rls = np.zeros(T, dtype=float)
        for k in range(T):
            C = R_tdl[k, :]
            d_true = float(d_full[k])
            e2_kal[k] = (d_true - float(C @ w_kal[:, k])) ** 2
            e2_rls[k] = (d_true - float(C @ w_rls[:, k])) ** 2

        ax2.plot(10.0 * np.log10(e2_kal + 1e-20), label="Kalman")
        ax2.plot(10.0 * np.log10(e2_rls + 1e-20), label="RLS", linestyle="--")
        ax2.axvline(train_end, linestyle="--")
        ax2.set_title(r"$10\log_{10}|e(k)|^2$")
        ax2.set_xlabel("iteration k")
        ax2.grid(True)
        ax2.legend()

        # (3) Coefficient evolution (w0)
        ax3.plot(w_kal[0, :], label="Kalman")
        ax3.plot(w_rls[0, :], label="RLS", linestyle="--")
        ax3.axvline(train_end, linestyle="--")
        ax3.set_title("Coefficient evolution: w0")
        ax3.set_xlabel("k")
        ax3.grid(True)
        ax3.legend()

        # (4) Coefficient evolution (w1 or last)
        idx = 1 if S > 1 else 0
        ax4.plot(w_kal[idx, :], label=f"Kalman w{idx}")
        ax4.plot(w_rls[idx, :], label=f"RLS w{idx}", linestyle="--")
        ax4.axvline(train_end, linestyle="--")
        ax4.set_title(f"Coefficient evolution: w{idx}")
        ax4.set_xlabel("k")
        ax4.grid(True)
        ax4.legend()

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
        "acc_err_kal": acc_err_kal,
        "acc_err_rls": acc_err_rls,
    }


if __name__ == "__main__":
    main(seed=0, plot=True)
