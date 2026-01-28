#  lms.sign_error.py
#
#       Implements the Sign-Error LMS algorithm for REAL valued data.
#       (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
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


class SignError(AdaptiveFilter):
    """
    Sign-Error LMS (real-valued).

    This is a sign-error LMS variant that replaces the error term by its sign:

        y[k] = w^T x_k
        e[k] = d[k] - y[k]
        w <- w + mu * sign(e[k]) * x_k

    Notes
    -----
    - Real-valued only: enforced by `ensure_real_signals`.
    - Uses the unified base API via `validate_input`.
    - Returns a priori error (computed before update).
    """

    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        step_size:
            Step-size (mu).
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.step_size = float(step_size)

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
        Run Sign-Error LMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k] (real).
        desired_signal:
            Desired signal d[k] (real).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the last sign(e[k]) value in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                Coefficient history stored in the base class.
            error_type:
                "a_priori".
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(x.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        x_padded = np.zeros(n_samples + m, dtype=np.float64)
        x_padded[m:] = x

        last_sign_e: Optional[float] = None

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]

            y_k = float(np.dot(self.w, x_k))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            s = float(np.sign(e_k))
            last_sign_e = s

            self.w = self.w + self.step_size * s * x_k
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[SignError] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"last_sign_error": last_sign_e}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF