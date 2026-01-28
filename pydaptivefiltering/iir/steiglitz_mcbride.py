# iir.steiglitz_mcbride.py
#
#       Implements the Steiglitz-McBride algorithm for REAL valued data.
#       (Algorithm 10.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, 3rd Ed., Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom            & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com      & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com            & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals


class SteiglitzMcBride(AdaptiveFilter):
    """
    Implements the Steiglitz-McBride (SM) algorithm for real-valued IIR adaptive filters.

    Notes
    -----
    This implementation follows the *classic* Steiglitz-McBride idea:
    - Build a prefiltered (approximately linear) regression using the current denominator estimate.
    - Update coefficients using the *filtered equation error* (auxiliary error).

    Coefficient vector convention:
    - First `poles_order` entries correspond to denominator (pole) parameters.
    - Remaining entries correspond to numerator (zero) parameters.
    """
    supports_complex: bool = False

    zeros_order: int
    poles_order: int
    step_size: float
    n_coeffs: int
    y_buffer: np.ndarray
    xf_buffer: np.ndarray
    df_buffer: np.ndarray

    def __init__(
        self,
        zeros_order: int,
        poles_order: int,
        step_size: float = 1e-3,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        zeros_order:
            Numerator order (number of zeros).
        poles_order:
            Denominator order (number of poles).
        step_size:
            Step size used in the coefficient update.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=zeros_order + poles_order, w_init=w_init)

        self.zeros_order = int(zeros_order)
        self.poles_order = int(poles_order)
        self.step_size = float(step_size)

        self.n_coeffs = int(self.zeros_order + 1 + self.poles_order)
        self.w = np.zeros(self.n_coeffs, dtype=np.float64)

        self.y_buffer = np.zeros(self.poles_order, dtype=np.float64)

        max_buffer: int = int(max(self.zeros_order + 1, self.poles_order + 1))
        self.xf_buffer = np.zeros(max_buffer, dtype=np.float64)
        self.df_buffer = np.zeros(max_buffer, dtype=np.float64)

    def _stability_procedure(self, a_coeffs: np.ndarray) -> np.ndarray:
        """
        Enforces IIR stability by reflecting poles outside the unit circle back inside.
        """
        poly_coeffs: np.ndarray = np.concatenate(([1.0], -a_coeffs))
        poles: np.ndarray = np.roots(poly_coeffs)
        mask: np.ndarray = np.abs(poles) > 1.0
        if np.any(mask):
            poles[mask] = 1.0 / np.conj(poles[mask])
            new_poly: np.ndarray = np.poly(poles)
            return -np.real(new_poly[1:])
        return a_coeffs

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
        Executes the Steiglitz-McBride adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the auxiliary (prefiltered) equation error in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Output computed from the current IIR model.
            errors:
                Output error e[k] = d[k] - y[k] (for evaluation/monitoring).
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_posteriori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["auxiliary_error"]:
            Filtered equation error sequence e_s[k] used in the SM update.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=np.float64)
        d: np.ndarray = np.asarray(desired_signal, dtype=np.float64)

        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors_s: np.ndarray = np.zeros(n_samples, dtype=np.float64)

        x_padded: np.ndarray = np.zeros(n_samples + self.zeros_order, dtype=np.float64)
        x_padded[self.zeros_order:] = x

        for k in range(n_samples):
            reg_x: np.ndarray = x_padded[k : k + self.zeros_order + 1][::-1]
            regressor: np.ndarray = np.concatenate((self.y_buffer, reg_x))

            y_k: float = float(np.dot(self.w, regressor))
            outputs[k] = y_k
            errors[k] = float(d[k] - y_k)

            a_coeffs: np.ndarray = self.w[: self.poles_order]

            xf_k: float = float(x[k] + np.dot(a_coeffs, self.xf_buffer[: self.poles_order]))
            df_k: float = float(d[k] + np.dot(a_coeffs, self.df_buffer[: self.poles_order]))

            self.xf_buffer = np.concatenate(([xf_k], self.xf_buffer[:-1]))
            self.df_buffer = np.concatenate(([df_k], self.df_buffer[:-1]))

            if self.poles_order == 0:
                regressor_s: np.ndarray = self.xf_buffer[: self.zeros_order + 1]
            else:
                regressor_s = np.concatenate(
                    (
                        self.df_buffer[1 : self.poles_order + 1],
                        self.xf_buffer[: self.zeros_order + 1],
                    )
                )

            e_s_k: float = float(df_k - np.dot(self.w, regressor_s))
            errors_s[k] = e_s_k

            self.w += 2.0 * self.step_size * regressor_s * e_s_k

            if self.poles_order > 0:
                self.w[: self.poles_order] = self._stability_procedure(self.w[: self.poles_order])
                self.y_buffer = np.concatenate(([y_k], self.y_buffer[:-1]))

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SteiglitzMcBride] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"auxiliary_error": errors_s}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_posteriori",
            extra=extra,
        )
# EOF