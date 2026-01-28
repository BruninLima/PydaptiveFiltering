# iir.error_equation.py
#
#       Implements the Equation Error RLS algorithm for REAL valued data.
#       (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, 3rd Ed., Diniz)
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


class ErrorEquation(AdaptiveFilter):
    """
    Implements the Equation Error RLS algorithm for real-valued IIR adaptive filtering.
    """
    supports_complex: bool = False

    zeros_order: int
    poles_order: int
    forgetting_factor: float
    epsilon: float
    n_coeffs: int
    Sd: np.ndarray
    y_buffer: np.ndarray
    d_buffer: np.ndarray

    def __init__(
        self,
        zeros_order: int,
        poles_order: int,
        forgetting_factor: float = 0.99,
        epsilon: float = 1e-3,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        zeros_order:
            Numerator order (number of zeros).
        poles_order:
            Denominator order (number of poles).
        forgetting_factor:
            Forgetting factor (lambda), typically close to 1.
        epsilon:
            Regularization / initialization parameter for the inverse correlation matrix.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.

        Notes
        -----
        Coefficient vector convention:
        - First `poles_order` entries correspond to denominator (pole) parameters.
        - Remaining entries correspond to numerator (zero) parameters.
        """
        super().__init__(filter_order=zeros_order + poles_order, w_init=w_init)

        self.zeros_order = int(zeros_order)
        self.poles_order = int(poles_order)
        self.forgetting_factor = float(forgetting_factor)
        self.epsilon = float(epsilon)

        self.n_coeffs = int(self.poles_order + 1 + self.zeros_order)
        self.w = np.zeros(self.n_coeffs, dtype=np.float64)

        self.Sd = (1.0 / self.epsilon) * np.eye(self.n_coeffs, dtype=np.float64)

        self.y_buffer = np.zeros(self.poles_order, dtype=np.float64)
        self.d_buffer = np.zeros(self.poles_order, dtype=np.float64)

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
        Executes the Equation Error RLS algorithm for IIR filters.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns pole coefficients trajectory in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k] computed from the output equation.
            errors:
                Output error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "equation_error".

        Extra (always)
        -------------
        extra["auxiliary_errors"]:
            Equation-error based auxiliary error sequence.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["a_coefficients"]:
            Trajectory of denominator (pole) coefficients, shape (N, poles_order).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=np.float64)
        d: np.ndarray = np.asarray(desired_signal, dtype=np.float64)

        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors_aux: np.ndarray = np.zeros(n_samples, dtype=np.float64)

        a_track: Optional[np.ndarray] = (
            np.zeros((n_samples, self.poles_order), dtype=np.float64)
            if (return_internal_states and self.poles_order > 0)
            else None
        )

        x_padded: np.ndarray = np.zeros(n_samples + self.zeros_order, dtype=np.float64)
        x_padded[self.zeros_order:] = x

        for k in range(n_samples):
            reg_x: np.ndarray = x_padded[k : k + self.zeros_order + 1][::-1]
            reg_y: np.ndarray = np.concatenate((self.y_buffer, reg_x))
            reg_e: np.ndarray = np.concatenate((self.d_buffer, reg_x))

            y_out: float = float(np.dot(self.w, reg_y))
            y_equation: float = float(np.dot(self.w, reg_e))

            outputs[k] = y_out
            errors[k] = float(d[k] - y_out)
            errors_aux[k] = float(d[k] - y_equation)

            psi: np.ndarray = self.Sd @ reg_e
            den: float = float(self.forgetting_factor + reg_e.T @ psi)

            self.Sd = (1.0 / self.forgetting_factor) * (self.Sd - np.outer(psi, psi) / den)
            self.w += (self.Sd @ reg_e) * errors_aux[k]

            if self.poles_order > 0:
                self.w[: self.poles_order] = self._stability_procedure(self.w[: self.poles_order])

                if return_internal_states and a_track is not None:
                    a_track[k, :] = self.w[: self.poles_order]

                self.y_buffer = np.concatenate(([y_out], self.y_buffer[:-1]))
                self.d_buffer = np.concatenate(([d[k]], self.d_buffer[:-1]))

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[ErrorEquation] Completed in {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "auxiliary_errors": errors_aux,
        }
        if return_internal_states:
            extra["a_coefficients"] = a_track

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="equation_error",
            extra=extra,
        )
# EOF