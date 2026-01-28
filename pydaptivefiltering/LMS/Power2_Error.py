#  lms.power2_error.py
#
#       Implements the Power-of-Two Error LMS algorithm for REAL valued data.
#       (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class Power2ErrorLMS(AdaptiveFilter):
    """
    Power-of-Two Error LMS (real-valued).

    This is an LMS variant where the instantaneous error is quantized to the nearest
    power-of-two (or special cases), aiming at reducing computational complexity.

    Quantization rule (as implemented here)
    ---------------------------------------
    Let e be the a priori error.

    - If |e| >= 1:
        q(e) = sign(e)
    - Else if |e| < 2^(-bd+1):
        q(e) = tau * sign(e)
    - Else:
        q(e) = 2^{floor(log2(|e|))} * sign(e)

    Update:
        w <- w + 2 * mu * q(e) * x_k

    Notes
    -----
    - Real-valued only: enforced by `ensure_real_signals`.
    - Uses the unified base API via `validate_input`.
    """

    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int,
        bd: int,
        tau: float,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        bd:
            Word length (signal bits) used in the small-error threshold 2^(-bd+1).
        tau:
            Gain factor used when |e| is very small (< 2^(-bd+1)).
        step_size:
            Step-size (mu).
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.bd = int(bd)
        self.tau = float(tau)
        self.step_size = float(step_size)

        if self.bd <= 0:
            raise ValueError(f"bd must be a positive integer. Got bd={self.bd}.")

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
        Run Power-of-Two Error LMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k] (real).
        desired_signal:
            Desired signal d[k] (real).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the last quantized error value in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output outputs[k].
            errors:
                A priori error errors[k] = d[k] - outputs[k].
            coefficients:
                History of coefficients stored in the base class.
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

        last_qe: Optional[float] = None
        small_thr = 2.0 ** (-self.bd + 1)

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]

            y_k = float(np.dot(self.w, x_k))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            abs_error = abs(e_k)
            if abs_error >= 1.0:
                qe = float(np.sign(e_k))
            elif abs_error < small_thr:
                qe = float(self.tau * np.sign(e_k))
            else:
                qe = float((2.0 ** np.floor(np.log2(abs_error))) * np.sign(e_k))

            last_qe = qe

            self.w = self.w + (2.0 * self.step_size) * qe * x_k
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[Power2ErrorLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"last_quantized_error": last_qe, "small_threshold": float(small_thr)}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF