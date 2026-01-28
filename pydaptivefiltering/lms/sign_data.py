#  lms.sign_data.py
#
#       Implements the Sign-Data LMS algorithm for COMPLEX valued data.
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

ArrayLike = Union[np.ndarray, list]


class SignData(AdaptiveFilter):
    """
    Sign-Data LMS (complex-valued).

    This is a low-complexity LMS variant where the input regressor is replaced by
    its elementwise sign:

        y[k] = w^H x_k
        e[k] = d[k] - y[k]
        w <- w + 2 * mu * conj(e[k]) * sign(x_k)

    Notes
    -----
    - Complex-valued implementation (supports_complex=True).
    - Uses the unified base API via `@validate_input`.
    - Returns a priori error by default (e[k] computed before update).
    """

    supports_complex: bool = True

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
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Sign-Data LMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the last regressor sign vector in result.extra.

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

        x = np.asarray(input_signal, dtype=complex).ravel()
        d = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(x.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        x_padded = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        last_sign_xk: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]

            y_k = complex(np.vdot(self.w, x_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            sign_xk = np.sign(x_k)
            last_sign_xk = sign_xk

            self.w = self.w + (2.0 * self.step_size) * np.conj(e_k) * sign_xk
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[SignData] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"last_sign_regressor": None if last_sign_xk is None else last_sign_xk.copy()}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF