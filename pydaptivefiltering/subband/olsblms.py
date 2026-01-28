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
from pydaptivefiltering.utils.validation import ensure_real_signals


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
    Implements the Open-Loop Subband LMS (OLSBLMS) adaptive-filtering algorithm for real-valued data.
    (Algorithm 12.1, Diniz)

    Notes
    -----
    - The adaptive coefficients are subband-wise: w has shape (M, Nw+1).
    - For compatibility with the base class, `OptimizationResult.coefficients` will contain
      a flattened history of the subband coefficient matrix (row-major flatten).
      The full matrix history is provided in `extra["w_matrix_history"]`.
    - The MATLAB reference typically evaluates MSE in subbands; here we also provide a
      convenience fullband reconstruction via the synthesis bank.
    """
    supports_complex: bool = False

    M: int
    Nw: int
    L: int
    step: float
    gamma: float
    a: float

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
        self.M = int(n_subbands)
        if self.M <= 0:
            raise ValueError("n_subbands must be a positive integer.")

        self.Nw = int(filter_order)
        if self.Nw <= 0:
            raise ValueError("filter_order must be a positive integer.")

        self.step = float(step)
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
        Executes the adaptation process for the Open-Loop Subband LMS (OLSBLMS) algorithm.

        Returns
        -------
        OptimizationResult
            outputs:
                Fullband reconstructed output y(n), same length as desired_signal.
            errors:
                Fullband output error e(n) = d(n) - y(n).
            coefficients:
                Flattened coefficient history (shape: (#snapshots, M*(Nw+1))).

        Extra (always)
        -------------
        extra["w_matrix_history"]:
            List of coefficient matrices (M, Nw+1), one per subband-iteration.
        extra["subband_outputs"], extra["subband_errors"]:
            Arrays with shape (M, N_iter).
        extra["mse_subbands"], extra["mse_overall"]:
            MSE curves in subbands.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["sig_ol"]:
            Final subband energy estimates (M,).
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

                mu_m = (2.0 * self.step) / (self.gamma + (self.Nw + 1) * sig_ol[m])

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
