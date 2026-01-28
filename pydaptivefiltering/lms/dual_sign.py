#  lms.dual_sign.py
#
#       Implements the DualSign LMS algorithm for REAL valued data.
#       (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms
#                                              and Practical Implementation, Diniz)
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
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class DualSign(AdaptiveFilter):
    """
    DualSign LMS (real-valued).

    This is a sign-error LMS variant that switches between two effective gains
    depending on the error magnitude, controlled by a threshold `rho`.

    Notes
    -----
    - Real-valued only: enforced by `@ensure_real_signals`.
    - Uses the unified base API via `@validate_input`:
        * optimize(input_signal=..., desired_signal=...)
        * optimize(x=..., d=...)
        * optimize(x, d)

    Update rule (one common form)
    -----------------------------
        e[k] = d[k] - y[k]
        u[k] = sign(e[k])              if |e[k]| <= rho
             = gamma * sign(e[k])      if |e[k]| >  rho
        w <- w + 2 * mu * u[k] * x_k
    """

    supports_complex: bool = False

    rho: float
    gamma: float
    step_size: float

    def __init__(
        self,
        filter_order: int,
        rho: float,
        gamma: float,
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
        rho:
            Threshold on |e[k]| that selects which sign gain is used.
        gamma:
            Gain multiplier used when |e[k]| > rho. Typically an integer > 1.
        step:
            Step-size (mu).
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        safe_eps:
            Small epsilon for internal safety checks (kept for consistency across the library).
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.step_size = float(step)
        self._safe_eps = float(safe_eps)

    @validate_input
    @ensure_real_signals
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run DualSign LMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k] (real).
        desired_signal:
            Desired signal d[k] (real).
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
                Coefficient history stored in the base class.
            error_type:
                "a_priori".
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=np.float64).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples: int = int(x.size)
        m: int = int(self.filter_order)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.float64)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=np.float64)
        x_padded[m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + m + 1][::-1]

            y_k: float = float(np.dot(self.w, x_k))
            outputs[k] = y_k

            e_k: float = float(d[k] - y_k)
            errors[k] = e_k

            s: float = float(np.sign(e_k))
            if abs(e_k) > self.rho:
                s *= self.gamma

            self.w = self.w + (2.0 * self.step_size) * s * x_k
            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[DualSign] Completed in {runtime_s * 1000:.03f} ms")

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
        )
# EOF