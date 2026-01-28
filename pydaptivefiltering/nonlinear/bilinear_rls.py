#  nonlinear.bilinear_rls.py
#
#       Implements the Bilinear RLS algorithm for REAL valued data.
#       (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class BilinearRLS(AdaptiveFilter):
    """
    Bilinear RLS (real-valued).

    Implements a bilinear regressor RLS structure (Algorithm 11.3 - Diniz).
    The regressor used here has 4 components:

        u[k] = [ x[k],
                 d[k-1],
                 x[k] d[k-1],
                 x[k-1] d[k-1] ]^T

    and the RLS update (a priori form) is:

        y[k] = w^T u[k]
        e[k] = d[k] - y[k]
        k[k] = P[k-1] u[k] / (lambda + u[k]^T P[k-1] u[k])
        P[k] = (P[k-1] - k[k] u[k]^T P[k-1]) / lambda
        w[k] = w[k-1] + k[k] e[k]

    Notes
    -----
    - Real-valued only: enforced by `ensure_real_signals`.
    - Uses the unified base API via `validate_input`.
    - Returns a priori error by default.
    """

    supports_complex: bool = False

    def __init__(
        self,
        forgetting_factor: float = 0.98,
        delta: float = 1.0,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        forgetting_factor:
            Forgetting factor lambda (0 < lambda <= 1).
        delta:
            Regularization factor for initial P = I/delta (delta > 0).
        w_init:
            Optional initial coefficients (length 4). If None, zeros.
        safe_eps:
            Small epsilon used to guard denominators.
        """
        n_coeffs = 4
        super().__init__(filter_order=n_coeffs - 1, w_init=w_init)

        self.lambda_factor = float(forgetting_factor)
        if not (0.0 < self.lambda_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must satisfy 0 < forgetting_factor <= 1. Got {self.lambda_factor}."
            )

        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError(f"delta must be > 0. Got delta={self.delta}.")

        self._safe_eps = float(safe_eps)

        self.P = np.eye(n_coeffs, dtype=np.float64) / self.delta

    @validate_input
    @ensure_real_signals
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Bilinear RLS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k] (real).
        desired_signal:
            Desired signal d[k] (real).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the last regressor and last gain in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k] (a priori).
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class (length 4).
            error_type:
                "a_priori".
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(x.size)
        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        x_prev = 0.0
        d_prev = 0.0

        last_u: Optional[np.ndarray] = None
        last_k: Optional[np.ndarray] = None

        for k in range(n_samples):
            u = np.array(
                [x[k], d_prev, x[k] * d_prev, x_prev * d_prev],
                dtype=np.float64,
            )
            last_u = u

            y_k = float(np.dot(self.w, u))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            Pu = self.P @ u
            denom = float(self.lambda_factor + (u @ Pu))
            if abs(denom) < self._safe_eps:
                denom = float(np.sign(denom) * self._safe_eps) if denom != 0.0 else float(self._safe_eps)

            k_gain = Pu / denom
            last_k = k_gain

            self.P = (self.P - np.outer(k_gain, Pu)) / self.lambda_factor

            self.w = self.w + k_gain * e_k
            self._record_history()

            x_prev = float(x[k])
            d_prev = float(d[k])

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[BilinearRLS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "P_last": self.P.copy(),
                "last_regressor": None if last_u is None else last_u.copy(),
                "last_gain": None if last_k is None else last_k.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF