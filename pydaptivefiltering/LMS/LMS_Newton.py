#  lms.lms_newton.py
#
#       Implements the Complex LMS-Newton algorithm for COMPLEX valued data.
#       (Algorithm 4.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class LMSNewton(AdaptiveFilter):
    """
    LMS-Newton (complex-valued).

    This algorithm approximates a Newton step by maintaining a recursive estimate of
    the inverse input correlation matrix, which tends to accelerate convergence in
    correlated-input scenarios.

    Notes
    -----
    - Complex-valued implementation (supports_complex = True).
    - Uses the unified base API via `@validate_input`:
        * optimize(input_signal=..., desired_signal=...)
        * optimize(x=..., d=...)
        * optimize(x, d)

    Recursion (one common form)
    ---------------------------
    Let P[k] approximate R_x^{-1}. With forgetting factor alpha (0 < alpha < 1),
    and regressor x_k (shape (M+1,)), define:

        phi = x_k^H P x_k
        denom = (1-alpha)/alpha + phi
        P <- (P - (P x_k x_k^H P)/denom) / (1-alpha)
        w <- w + mu * conj(e[k]) * (P x_k)

    where e[k] = d[k] - w^H x_k.
    """

    supports_complex: bool = True

    alpha: float
    step_size: float
    inv_rx: np.ndarray

    def __init__(
        self,
        filter_order: int,
        alpha: float,
        initial_inv_rx: np.ndarray,
        step: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        alpha:
            Forgetting factor (0 < alpha < 1).
        initial_inv_rx:
            Initial inverse correlation matrix P[0], shape (M+1, M+1).
        step:
            Step-size mu.
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        safe_eps:
            Small epsilon used to guard denominators.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.alpha = float(alpha)
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must satisfy 0 < alpha < 1. Got alpha={self.alpha}.")

        P0 = np.asarray(initial_inv_rx, dtype=complex)
        n_taps = int(filter_order) + 1
        if P0.shape != (n_taps, n_taps):
            raise ValueError(
                f"initial_inv_rx must have shape {(n_taps, n_taps)}. Got {P0.shape}."
            )
        self.inv_rx = P0

        self.step_size = float(step)
        self._safe_eps = float(safe_eps)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run LMS-Newton adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_priori".
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)
        m: int = int(self.filter_order)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + m + 1][::-1]

            y_k: complex = complex(np.vdot(self.w, x_k))
            outputs[k] = y_k

            e_k: complex = d[k] - y_k
            errors[k] = e_k

            x_col: np.ndarray = x_k.reshape(-1, 1)
            Px: np.ndarray = self.inv_rx @ x_col
            phi: complex = (x_col.conj().T @ Px).item()

            denom: complex = ((1.0 - self.alpha) / self.alpha) + phi
            if abs(denom) < self._safe_eps:
                denom = denom + (self._safe_eps + 0.0j)

            self.inv_rx = (self.inv_rx - (Px @ Px.conj().T) / denom) / (1.0 - self.alpha)

            self.w = self.w + self.step_size * np.conj(e_k) * Px.ravel()

            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[LMSNewton] Completed in {runtime_s * 1000:.03f} ms")

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
        )
# EOF