#  subband.olsblms.py
#
#       Implements the Open-Loop Subband (LMS) Adaptive-Filtering Algorithm for REAL valued data.
#       (Algorithm 12.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.validation import ensure_real_signals


ArrayLike = Union[np.ndarray, list]


def _fir_filter_causal(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal FIR filtering via convolution, equivalent to MATLAB's filter(h,1,x)."""
    h = np.asarray(h, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y_full = np.convolve(x, h, mode="full")
    return y_full[: x.size]


def _decimate_by_L(x: np.ndarray, L: int) -> np.ndarray:
    """Decimate by L keeping samples 0, L, 2L, ... (MATLAB mod-index selection)."""
    return x[::L]


def _upsample_by_L(x: np.ndarray, L: int, out_len: int) -> np.ndarray:
    """Zero-stuffing upsample: place x[k] at y[k*L], zeros elsewhere, truncated to out_len."""
    y = np.zeros((out_len,), dtype=float)
    max_k = min(x.size, (out_len + L - 1) // L)
    y[: max_k * L : L] = x[:max_k]
    return y


class OLSBLMS(AdaptiveFilter):
    """
    Open-Loop Subband LMS (OLSBLMS) for real-valued fullband signals.

    Implements the Open-Loop Subband LMS adaptive filtering algorithm
    (Algorithm 12.1, Diniz) using an analysis/synthesis filterbank with
    subband-adaptive FIR filters.

    High-level operation (as implemented)
    -------------------------------------
    Given fullband input x[n] and desired d[n], and an M-channel analysis bank h_k[m],
    the algorithm proceeds in two stages:

    (A) Analysis + Decimation (open-loop)
        For each subband m = 0..M-1:
          - Filter the fullband input and desired with the analysis filter:
                x_aux[m] = filter(hk[m], 1, x)
                d_aux[m] = filter(hk[m], 1, d)
          - Decimate by L (keep samples 0, L, 2L, ...):
                x_sb[m] = x_aux[m][::L]
                d_sb[m] = d_aux[m][::L]

        The adaptation length is:
            N_iter = min_m len(x_sb[m]) and len(d_sb[m])
        (i.e., all subbands are truncated to the shortest decimated sequence).

    (B) Subband LMS adaptation (per-sample in decimated time)
        Each subband has its own tapped-delay line x_ol[m,:] of length (Nw+1) and
        its own coefficient vector w_mat[m,:] (also length Nw+1).

        For each decimated-time index k = 0..N_iter-1, and for each subband m:
          - Update subband delay line:
                x_ol[m,0] = x_sb[m,k]
          - Compute subband output and error:
                y_sb[m,k] = w_mat[m]^T x_ol[m]
                e_sb[m,k] = d_sb[m,k] - y_sb[m,k]
          - Update a smoothed subband energy estimate:
                sig_ol[m] = (1-a) sig_ol[m] + a * x_sb[m,k]^2
          - Normalized LMS-like step:
                mu_m = (2*step) / (gamma + (Nw+1)*sig_ol[m])
          - Coefficient update:
                w_mat[m] <- w_mat[m] + mu_m * e_sb[m,k] * x_ol[m]

    Fullband reconstruction (convenience synthesis)
    ----------------------------------------------
    After adaptation, a fullband output is reconstructed via the synthesis bank f_k[m]:
      - Upsample each subband output by L (zero-stuffing), then filter:
            y_up[m]   = upsample(y_sb[m], L)
            y_full[m] = filter(fk[m], 1, y_up[m])
      - Sum across subbands:
            y[n] = sum_m y_full[m][n]
    The returned error is the fullband output error e[n] = d[n] - y[n].

    Coefficient representation and history
    --------------------------------------
    - The adaptive parameters are stored as:
          w_mat : ndarray, shape (M, Nw+1), dtype=float
    - For compatibility with the base class, `self.w` is a flattened view of w_mat
      (row-major), and `OptimizationResult.coefficients` contains the stacked history
      of this flattened vector (recorded once per decimated-time iteration, plus the
      initial entry).
    - The full (M, Nw+1) snapshots are also stored in `extra["w_matrix_history"]`.

    Parameters
    ----------
    n_subbands : int
        Number of subbands (M).
    analysis_filters : array_like
        Analysis bank hk with shape (M, Lh).
    synthesis_filters : array_like
        Synthesis bank fk with shape (M, Lf).
    filter_order : int
        Subband FIR order Nw (number of taps per subband is Nw+1).
    step_size : float, default=0.1
        Global LMS step-size factor.
    gamma : float, default=1e-2
        Regularization term in the normalized denominator (>0 recommended).
    a : float, default=0.01
        Exponential smoothing factor for subband energy estimates in (0,1].
    decimation_factor : int, optional
        Decimation factor L. If None, uses L=M.
    w_init : array_like, optional
        Initial subband coefficients. Can be:
          - shape (M, Nw+1), or
          - flat of length M*(Nw+1), reshaped row-major.

    Notes
    -----
    - Real-valued interface (input_signal and desired_signal enforced real).
    - This is an *open-loop* structure: subband regressors are formed from the
      analysis-filtered fullband input, independent of any reconstructed fullband
      output loop.
    - Subband MSE curves are provided as `mse_subbands = e_sb**2` and
      `mse_overall = mean_m mse_subbands[m,k]`.

    """
    supports_complex: bool = False

    M: int
    Nw: int
    L: int
    step_size: float
    gamma: float
    a: float

    def __init__(
        self,
        n_subbands: int,
        analysis_filters: ArrayLike,
        synthesis_filters: ArrayLike,
        filter_order: int,
        step_size: float = 0.1,
        gamma: float = 1e-2,
        a: float = 0.01,
        decimation_factor: Optional[int] = None,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        self.M = int(n_subbands)
        if self.M <= 0:
            raise ValueError("n_subbands must be a positive integer.")

        self.Nw = int(filter_order)
        if self.Nw <= 0:
            raise ValueError("filter_order must be a positive integer.")

        self.step_size = float(step_size)
        self.gamma = float(gamma)
        self.a = float(a)

        hk = np.asarray(analysis_filters, dtype=float)
        fk = np.asarray(synthesis_filters, dtype=float)

        if hk.ndim != 2 or fk.ndim != 2:
            raise ValueError("analysis_filters and synthesis_filters must be 2D arrays with shape (M, Lh/Lf).")
        if hk.shape[0] != self.M or fk.shape[0] != self.M:
            raise ValueError(
                f"Filterbanks must have M rows. Got hk.shape[0]={hk.shape[0]}, fk.shape[0]={fk.shape[0]}, M={self.M}."
            )

        self.hk = hk
        self.fk = fk

        self.L = int(decimation_factor) if decimation_factor is not None else self.M
        if self.L <= 0:
            raise ValueError("decimation_factor L must be a positive integer.")

        self._n_params = int(self.M * (self.Nw + 1))
        super().__init__(filter_order=self._n_params - 1, w_init=None)

        self.w_mat: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=float)
        if w_init is not None:
            w0 = np.asarray(w_init, dtype=float)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.w_mat = w0.copy()
            elif w0.ndim == 1 and w0.size == self._n_params:
                self.w_mat = w0.reshape(self.M, self.Nw + 1).copy()
            else:
                raise ValueError(
                    "w_init must have shape (M, Nw+1) or be a flat vector of length M*(Nw+1). "
                    f"Got w_init.shape={w0.shape}."
                )

        self.w = self.w_mat.reshape(-1).astype(float, copy=False)
        self.w_history = []
        self._record_history()

        self.w_matrix_history: list[np.ndarray] = []

    def _sync_base_w(self) -> None:
        """Keep base `self.w` consistent with the subband matrix."""
        self.w = self.w_mat.reshape(-1).astype(float, copy=False)

    @classmethod
    def default_test_init_kwargs(cls, order: int) -> dict:
        M = 1
        hk = np.array([[1.0]], dtype=float)
        fk = np.array([[1.0]], dtype=float)
        return dict(
            n_subbands=M,
            analysis_filters=hk,
            synthesis_filters=fk,
            filter_order=order,
            step=0.1,
            gamma=1e-2,
            a=0.01,
            decimation_factor=1,
        )

    @ensure_real_signals
    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run OLSBLMS adaptation.

        Parameters
        ----------
        input_signal : array_like of float
            Fullband input x[n], shape (N,).
        desired_signal : array_like of float
            Fullband desired d[n], shape (N,).
        verbose : bool, default=False
            If True, prints runtime and iteration count.
        return_internal_states : bool, default=False
            If True, returns additional internal states in result.extra.

        Returns
        -------
        OptimizationResult
            outputs : ndarray of float, shape (N,)
                Fullband reconstructed output y[n] obtained by synthesis of the
                subband outputs after adaptation.
            errors : ndarray of float, shape (N,)
                Fullband output error e[n] = d[n] - y[n].
            coefficients : ndarray
                Flattened coefficient history of w_mat, shape
                (#snapshots, M*(Nw+1)), where snapshots are recorded once per
                subband-iteration (decimated-time step), plus the initial entry.
            error_type : str
                "output_error".

            extra : dict
                Always contains:
                  - "w_matrix_history": list of (M, Nw+1) coefficient snapshots
                  - "subband_outputs": ndarray (M, N_iter)
                  - "subband_errors": ndarray (M, N_iter)
                  - "mse_subbands": ndarray (M, N_iter) with e_sb**2
                  - "mse_overall": ndarray (N_iter,) mean subband MSE per iteration
                If return_internal_states=True, also contains:
                  - "sig_ol": final subband energy estimates, shape (M,)
        """
        tic: float = time()

        x = np.asarray(input_signal, dtype=float).ravel()
        d = np.asarray(desired_signal, dtype=float).ravel()

        n_samples: int = int(x.size)

        xsb_list: list[np.ndarray] = []
        dsb_list: list[np.ndarray] = []
        for m in range(self.M):
            xaux_x = _fir_filter_causal(self.hk[m, :], x)
            xaux_d = _fir_filter_causal(self.hk[m, :], d)
            xsb_list.append(_decimate_by_L(xaux_x, self.L))
            dsb_list.append(_decimate_by_L(xaux_d, self.L))

        N_iter: int = min(arr.size for arr in (xsb_list + dsb_list)) if (xsb_list and dsb_list) else 0
        if N_iter == 0:
            y0 = np.zeros_like(d)
            runtime_s = float(time() - tic)
            return self._pack_results(
                outputs=y0,
                errors=d - y0,
                runtime_s=runtime_s,
                error_type="output_error",
                extra={
                    "w_matrix_history": [],
                    "subband_outputs": np.zeros((self.M, 0), dtype=float),
                    "subband_errors": np.zeros((self.M, 0), dtype=float),
                    "mse_subbands": np.zeros((self.M, 0), dtype=float),
                    "mse_overall": np.zeros((0,), dtype=float),
                },
            )

        xsb = np.vstack([arr[:N_iter] for arr in xsb_list])
        dsb = np.vstack([arr[:N_iter] for arr in dsb_list])

        y_sb = np.zeros((self.M, N_iter), dtype=float)
        e_sb = np.zeros((self.M, N_iter), dtype=float)

        x_ol = np.zeros((self.M, self.Nw + 1), dtype=float)
        sig_ol = np.zeros((self.M,), dtype=float)

        self.w_history = []
        self._record_history()
        self.w_matrix_history = []

        for k in range(N_iter):
            for m in range(self.M):
                x_ol[m, 1:] = x_ol[m, :-1]
                x_ol[m, 0] = xsb[m, k]

                y_sb[m, k] = float(np.dot(self.w_mat[m, :], x_ol[m, :]))
                e_sb[m, k] = float(dsb[m, k] - y_sb[m, k])

                sig_ol[m] = (1.0 - self.a) * sig_ol[m] + self.a * (xsb[m, k] ** 2)

                mu_m = (2.0 * self.step_size) / (self.gamma + (self.Nw + 1) * sig_ol[m])

                self.w_mat[m, :] = self.w_mat[m, :] + mu_m * e_sb[m, k] * x_ol[m, :]

            self.w_matrix_history.append(self.w_mat.copy())
            self._sync_base_w()
            self._record_history()

        y_full = np.zeros((n_samples,), dtype=float)
        for m in range(self.M):
            y_up = _upsample_by_L(y_sb[m, :], self.L, n_samples)
            y_full += _fir_filter_causal(self.fk[m, :], y_up)

        e_full = d - y_full

        mse_subbands = e_sb ** 2
        mse_overall = np.mean(mse_subbands, axis=0)

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[OLSBLMS] Completed in {runtime_s * 1000:.03f} ms | iters={N_iter}")

        extra: Dict[str, Any] = {
            "w_matrix_history": self.w_matrix_history,
            "subband_outputs": y_sb,
            "subband_errors": e_sb,
            "mse_subbands": mse_subbands,
            "mse_overall": mse_overall,
        }
        if return_internal_states:
            extra["sig_ol"] = sig_ol.copy()

        return self._pack_results(
            outputs=y_full,
            errors=e_full,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF
