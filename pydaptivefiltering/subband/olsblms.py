#  subband.olsblms.py
#
#       Implements the Open-Loop Subband (LMS) Adaptive-Filtering Algorithm for REAL valued data.
#       (Algorithm 12.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
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
from typing import Optional, Union, List, Dict, Any

from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals


ArrayLike = Union[np.ndarray, list]


def _fir_filter_causal(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Causal FIR filtering via convolution, equivalent to MATLAB's filter(h,1,x).
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y_full = np.convolve(x, h, mode="full")
    return y_full[: x.size]


def _decimate_by_L(x: np.ndarray, L: int) -> np.ndarray:
    """
    MATLAB:
        x_dec = xaux(find(mod((1:length(xaux))-1,L)==0));

    Which, in 0-based indexing, is:
        x_dec = xaux[0], xaux[L], xaux[2L], ...
    """
    return x[::L]


def _upsample_by_L(x: np.ndarray, L: int, out_len: int) -> np.ndarray:
    """
    Zero-stuffing upsample: place x[k] at y[k*L], zeros elsewhere, truncated to out_len.
    """
    y = np.zeros((out_len,), dtype=float)
    max_k = min(x.size, (out_len + L - 1) // L)
    y[: max_k * L : L] = x[:max_k]
    return y


class OLSBLMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Open-Loop Subband LMS (OLSBLMS) adaptive-filtering algorithm
        for REAL-valued data.

          1) Analysis:
                xsb(m,k) = (hk(m,:) * x)[kL]     (FIR analysis + decimation by L)
                dsb(m,k) = (hk(m,:) * d)[kL]
             with L = M by default.

          2) Open-loop LMS adaptation in each subband m:
                x_ol(m,:) <- [xsb(m,k), x_ol(m,1:Nw)]
                e_ol(m,k) = dsb(m,k) - w_ol(m,:)^T x_ol(m,:)
                sig_ol(m) <- (1-a)*sig_ol(m) + a*(xsb(m,k))^2
                mu_m(k)   = 2*u / (gamma + (Nw+1)*sig_ol(m))
                w_ol(m,:) <- w_ol(m,:) + mu_m(k)*e_ol(m,k)*x_ol(m,:)

          3) (Optional for API consistency) Synthesis:
             The MATLAB script computes MSE directly from subband errors and does not
             reconstruct a fullband output y(n). For convenience and consistency with
             the library API, this implementation reconstructs a fullband estimate by:
                - Upsampling each subband output by L
                - Filtering with the corresponding synthesis filter fk(m,:)
                - Summing across subbands

    Notes
    -----
        - This algorithm is intended for REAL-valued signals. Complex is not supported.
        - The provided analysis/synthesis filterbanks should be designed as a PR (perfect
          reconstruction) pair for best fullband reconstruction quality (as in the toolbox).

    Parameters
    ----------
        n_subbands : int
            Number of subbands M (must match hk.shape[0] and fk.shape[0]).
        analysis_filters : array_like
            Analysis filterbank hk with shape (M, Lh). Each row is one FIR analysis filter.
        synthesis_filters : array_like
            Synthesis filterbank fk with shape (M, Lf). Each row is one FIR synthesis filter.
        filter_order : int
            Order of each adaptive filter in the subbands (Nw). Taps per subband = Nw+1.
        step : float, default=0.1
            Convergence factor u (MATLAB variable u).
        gamma : float, default=1e-2
            Regularization term to avoid large normalized steps (MATLAB variable gamma).
        a : float, default=0.01
            Exponential averaging factor for subband energy (MATLAB variable a).
        decimation_factor : int | None, default=None
            Decimation/interpolation factor L. If None, defaults to L=M (MATLAB: L=M).
        w_init : array_like | None
            Optional initial coefficients. Accepts:
              - shape (M, Nw+1) : one coefficient vector per subband
              - shape (M*(Nw+1),) : flattened; will be reshaped to (M, Nw+1)

    Attributes
    ----------
        supports_complex : bool
            False (REAL-valued algorithm).
        w : np.ndarray
            Current subband coefficient matrix with shape (M, Nw+1).
    """

    supports_complex: bool = False

    def __init__(
        self,
        n_subbands: int,
        analysis_filters: ArrayLike,
        synthesis_filters: ArrayLike,
        filter_order: int,
        step: float = 0.1,
        gamma: float = 1e-2,
        a: float = 0.01,
        decimation_factor: Optional[int] = None,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        # Base class is used for input validation helpers & common metadata.
        super().__init__(filter_order, w_init=None)

        self.M: int = int(n_subbands)
        if self.M <= 0:
            raise ValueError("n_subbands must be a positive integer.")

        self.Nw: int = int(filter_order)
        if self.Nw <= 0:
            raise ValueError("filter_order must be a positive integer.")

        self.step: float = float(step)
        self.gamma: float = float(gamma)
        self.a: float = float(a)

        hk = np.asarray(analysis_filters, dtype=float)
        fk = np.asarray(synthesis_filters, dtype=float)

        if hk.ndim != 2 or fk.ndim != 2:
            raise ValueError("analysis_filters and synthesis_filters must be 2D arrays with shape (M, Lh/Lf).")
        if hk.shape[0] != self.M or fk.shape[0] != self.M:
            raise ValueError(
                f"Filterbanks must have M rows. Got hk.shape[0]={hk.shape[0]}, fk.shape[0]={fk.shape[0]}, M={self.M}."
            )

        self.hk: np.ndarray = hk
        self.fk: np.ndarray = fk

        self.L: int = int(decimation_factor) if decimation_factor is not None else self.M
        if self.L <= 0:
            raise ValueError("decimation_factor L must be a positive integer.")

        # Subband coefficients (M x (Nw+1))
        self.w: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=float)

        if w_init is not None:
            w0 = np.asarray(w_init, dtype=float)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.w = w0.copy()
            elif w0.ndim == 1 and w0.size == self.M * (self.Nw + 1):
                self.w = w0.reshape(self.M, self.Nw + 1).copy()
            else:
                raise ValueError(
                    "w_init must have shape (M, Nw+1) or be a flat vector of length M*(Nw+1). "
                    f"Got w_init.shape={w0.shape}."
                )

        # Local history for 2D coefficients
        self.w_history: List[np.ndarray] = []

    def _record_history(self) -> None:
        # Store the full (M, Nw+1) matrix at each iteration (like wk_ol in MATLAB)
        self.w_history.append(self.w.copy())

    @ensure_real_signals
    def optimize(
        self,
        input_signal: ArrayLike,
        desired_signal: ArrayLike,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Description
        -----------
            Executes the adaptation process for the Open-Loop Subband LMS (OLSBLMS) algorithm.

        Inputs
        -------
            input_signal : np.ndarray | list
                Fullband input signal x(n) (REAL-valued).
            desired_signal : np.ndarray | list
                Fullband desired signal d(n) (REAL-valued).
            verbose : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs : np.ndarray
                    Fullband reconstructed output y(n), same length as desired_signal.
                    (Note: reconstruction is an extra convenience beyond the MATLAB script.)
                errors : np.ndarray
                    Fullband error e(n) = d(n) - y(n), same length as desired_signal.
                coefficients : list[np.ndarray]
                    History of subband coefficient matrices, each of shape (M, Nw+1),
                    stored once per subband iteration k.
                subband_outputs : np.ndarray
                    Subband outputs y_sb with shape (M, N_iter).
                subband_errors : np.ndarray
                    Subband errors e_sb with shape (M, N_iter).
                mse_subbands : np.ndarray
                    MSE per subband over time, shape (M, N_iter) (instantaneous e^2 averaged across runs in MATLAB).
                mse_overall : np.ndarray
                    Overall MSE curve (mean over subbands), shape (N_iter,).

        Main Variables (MATLAB names)
        -----------------------------
            hk, fk : analysis/synthesis filterbanks
            L      : decimation/interpolation factor (default L=M)
            xsb    : subband input signals after analysis + decimation
            dsb    : subband desired signals after analysis + decimation
            w_ol   : subband adaptive coefficient vectors
            x_ol   : subband tapped-delay vectors
            sig_ol : subband energy estimates
            e_ol   : subband errors

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
        """
        tic: float = time()

        x = np.asarray(input_signal, dtype=float).reshape(-1)
        d = np.asarray(desired_signal, dtype=float).reshape(-1)

        self._validate_inputs(x, d)
        n_samples: int = int(x.size)

        # ==========================================================
        # 1) Analysis of desired and input signals (hk + decimation)
        # ==========================================================
        xsb_list: List[np.ndarray] = []
        dsb_list: List[np.ndarray] = []
        for m in range(self.M):
            xaux_x = _fir_filter_causal(self.hk[m, :], x)
            xaux_d = _fir_filter_causal(self.hk[m, :], d)
            xsb_list.append(_decimate_by_L(xaux_x, self.L))
            dsb_list.append(_decimate_by_L(xaux_d, self.L))

        # same N across subbands (MATLAB uses N iterations explicitly; here infer from data)
        N_iter: int = min(arr.size for arr in xsb_list + dsb_list)
        if N_iter == 0:
            y0 = np.zeros_like(d)
            return {
                "outputs": y0,
                "errors": d - y0,
                "coefficients": [],
                "subband_outputs": np.zeros((self.M, 0), dtype=float),
                "subband_errors": np.zeros((self.M, 0), dtype=float),
                "mse_subbands": np.zeros((self.M, 0), dtype=float),
                "mse_overall": np.zeros((0,), dtype=float),
            }

        xsb = np.vstack([arr[:N_iter] for arr in xsb_list])  # (M, N_iter)
        dsb = np.vstack([arr[:N_iter] for arr in dsb_list])  # (M, N_iter)

        # ==========================================================
        # 2) Subband open-loop adaptation (per MATLAB)
        # ==========================================================
        y_sb = np.zeros((self.M, N_iter), dtype=float)
        e_sb = np.zeros((self.M, N_iter), dtype=float)

        x_ol = np.zeros((self.M, self.Nw + 1), dtype=float)
        sig_ol = np.zeros((self.M,), dtype=float)

        self.w_history = []

        for k in range(N_iter):
            for m in range(self.M):
                # x_ol(m,:) = [xsb(m,k) x_ol(m,1:Nw)]
                x_ol[m, 1:] = x_ol[m, :-1]
                x_ol[m, 0] = xsb[m, k]

                # e_ol = dsb - w^T x
                y_sb[m, k] = float(np.dot(self.w[m, :], x_ol[m, :]))
                e_sb[m, k] = float(dsb[m, k] - y_sb[m, k])

                # sig_ol(m) = (1-a)*sig_ol(m) + a*(xsb(m,k))^2
                sig_ol[m] = (1.0 - self.a) * sig_ol[m] + self.a * (xsb[m, k] ** 2)

                # unlms_ol = 2*u/(gamma+(Nw+1)*sig_ol(m))
                mu_m = (2.0 * self.step) / (self.gamma + (self.Nw + 1) * sig_ol[m])

                # w_ol = w_ol + mu_m * e_ol * x_ol
                self.w[m, :] = self.w[m, :] + mu_m * e_sb[m, k] * x_ol[m, :]

            self._record_history()

        # ==========================================================
        # 3) (Convenience) Fullband reconstruction via synthesis bank
        # ==========================================================
        y_full = np.zeros((n_samples,), dtype=float)
        for m in range(self.M):
            y_up = _upsample_by_L(y_sb[m, :], self.L, n_samples)
            y_full += _fir_filter_causal(self.fk[m, :], y_up)

        e_full = d - y_full

        mse_sub = e_sb ** 2
        mse_overall = np.mean(mse_sub, axis=0)

        if verbose:
            print(f"OLSBLMS Adaptation completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": y_full,
            "errors": e_full,
            "coefficients": self.w_history,
            "subband_outputs": y_sb,
            "subband_errors": e_sb,
            "mse_subbands": mse_sub,
            "mse_overall": mse_overall,
        }
# EOF