# iir.rls_iir.py
#
#       Implements the RLS version of the Output Error algorithm (also known
#       as RLS adaptive IIR filter) for REAL valued data.
#       (Algorithm 10.1 - book: Adaptive Filtering: Algorithms and Practical
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


class RLSIIR(AdaptiveFilter):
    """
    Implements the RLS version of the Output Error algorithm for real-valued IIR adaptive filters.

    Notes
    -----
    Coefficient vector convention:
    - First `poles_order` entries correspond to denominator (pole) parameters.
    - Remaining entries correspond to numerator (zero) parameters.
    """
    supports_complex: bool = False

    zeros_order: int
    poles_order: int
    forgetting_factor: float
    delta: float
    n_coeffs: int
    Sd: np.ndarray
    y_buffer: np.ndarray
    x_line_buffer: np.ndarray
    y_line_buffer: np.ndarray

    def __init__(
        self,
        zeros_order: int,
        poles_order: int,
        forgetting_factor: float = 0.99,
        delta: float = 1e-3,
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
        delta:
            Regularization parameter used to initialize Sd (inverse covariance).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=zeros_order + poles_order, w_init=w_init)

        self.zeros_order = int(zeros_order)
        self.poles_order = int(poles_order)
        self.forgetting_factor = float(forgetting_factor)
        self.delta = float(delta)

        self.n_coeffs = int(self.zeros_order + self.poles_order + 1)
        self.w = np.zeros(self.n_coeffs, dtype=np.float64)

        self.Sd = (1.0 / self.delta) * np.eye(self.n_coeffs, dtype=np.float64)

        self.y_buffer = np.zeros(self.poles_order, dtype=np.float64)

        max_buffer: int = int(max(self.zeros_order + 1, self.poles_order))
        self.x_line_buffer = np.zeros(max_buffer, dtype=np.float64)
        self.y_line_buffer = np.zeros(max_buffer, dtype=np.float64)

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
        Executes the RLS-IIR (Output Error) adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns sensitivity signals in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                Output error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "output_error".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["x_sensitivity"]:
            Sensitivity-related track (x_line), length N.
        extra["y_sensitivity"]:
            Sensitivity-related track (y_line), length N.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=np.float64)
        d: np.ndarray = np.asarray(desired_signal, dtype=np.float64)

        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.float64)

        x_line_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=np.float64) if return_internal_states else None
        y_line_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=np.float64) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.zeros_order, dtype=np.float64)
        x_padded[self.zeros_order:] = x

        for k in range(n_samples):
            reg_x: np.ndarray = x_padded[k : k + self.zeros_order + 1][::-1]
            regressor: np.ndarray = np.concatenate((self.y_buffer, reg_x))

            y_k: float = float(np.dot(self.w, regressor))
            outputs[k] = y_k

            e_k: float = float(d[k] - y_k)
            errors[k] = e_k

            a_coeffs: np.ndarray = self.w[: self.poles_order]

            x_line_k: float = float(x[k] + np.dot(a_coeffs, self.x_line_buffer[: self.poles_order]))

            y_line_k: float = 0.0
            if self.poles_order > 0:
                prev_y: float = float(outputs[k - 1]) if k > 0 else 0.0
                y_line_k = float(-prev_y + np.dot(a_coeffs, self.y_line_buffer[: self.poles_order]))

            self.x_line_buffer = np.concatenate(([x_line_k], self.x_line_buffer[:-1]))
            self.y_line_buffer = np.concatenate(([y_line_k], self.y_line_buffer[:-1]))

            if return_internal_states and x_line_track is not None and y_line_track is not None:
                x_line_track[k] = x_line_k
                y_line_track[k] = y_line_k

            phi: np.ndarray = np.concatenate(
                (
                    self.y_line_buffer[: self.poles_order],
                    -self.x_line_buffer[: self.zeros_order + 1],
                )
            )

            psi: np.ndarray = self.Sd @ phi
            den: float = float(self.forgetting_factor + phi.T @ psi)
            self.Sd = (1.0 / self.forgetting_factor) * (self.Sd - np.outer(psi, psi) / den)

            self.w -= (self.Sd @ phi) * e_k

            if self.poles_order > 0:
                self.w[: self.poles_order] = self._stability_procedure(self.w[: self.poles_order])
                self.y_buffer = np.concatenate(([y_k], self.y_buffer[:-1]))

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[RLSIIR] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "x_sensitivity": x_line_track,
                "y_sensitivity": y_line_track,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF