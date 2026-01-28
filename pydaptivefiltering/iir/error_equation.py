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
from pydaptivefiltering._utils.validation import ensure_real_signals


class ErrorEquation(AdaptiveFilter):
    """
    Equation-Error RLS for adaptive IIR filtering (real-valued).

    The equation-error approach avoids the non-convexity of direct IIR
    output-error minimization by adapting the coefficients using an auxiliary
    (linear-in-parameters) error in which past outputs in the feedback path are
    replaced by past desired samples. This yields a quadratic (RLS-suitable)
    criterion while still producing a "true IIR" output for evaluation.

    This implementation follows Diniz (3rd ed., Alg. 10.3) and is restricted to
    **real-valued** signals (enforced by ``ensure_real_signals``).

    Parameters
    ----------
    zeros_order : int
        Numerator order ``N`` (number of zeros). The feedforward part has
        ``N + 1`` coefficients.
    poles_order : int
        Denominator order ``M`` (number of poles). The feedback part has ``M``
        coefficients.
    forgetting_factor : float, optional
        Exponential forgetting factor ``lambda``. Default is 0.99.
    epsilon : float, optional
        Positive initialization for the inverse correlation matrix used by RLS.
        Internally, the inverse covariance is initialized as:

        .. math::
            S(0) = \\frac{1}{\\epsilon} I.

        Default is 1e-3.
    w_init : array_like of float, optional
        Optional initial coefficient vector. If provided, it should have shape
        ``(M + N + 1,)`` following the parameter order described below. If None,
        the implementation initializes with zeros (and ignores ``w_init``).

    Notes
    -----
    Parameterization (as implemented)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The coefficient vector is arranged as:

    - ``w[:M]``: feedback (pole) coefficients (denoted ``a`` in literature)
    - ``w[M:]``: feedforward (zero) coefficients (denoted ``b``)

    Regressors and two outputs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    At time ``k``, define ``reg_x = [x(k), x(k-1), ..., x(k-N)]^T``.
    The algorithm forms two regressors:

    - Output regressor (uses past *true outputs*):

      .. math::
          \\varphi_y(k) = [y(k-1), \\ldots, y(k-M),\\; x(k), \\ldots, x(k-N)]^T.

    - Equation regressor (uses past *desired samples*):

      .. math::
          \\varphi_e(k) = [d(k-1), \\ldots, d(k-M),\\; x(k), \\ldots, x(k-N)]^T.

    The reported output is the "true IIR" output computed with the output
    regressor:

    .. math::
        y(k) = w^T(k)\\, \\varphi_y(k),

    while the auxiliary "equation" output is:

    .. math::
        y_{eq}(k) = w^T(k)\\, \\varphi_e(k).

    The adaptation is driven by the *equation error*:

    .. math::
        e_{eq}(k) = d(k) - y_{eq}(k),

    whereas the "output error" used for evaluation is:

    .. math::
        e(k) = d(k) - y(k).

    Stability procedure
    ~~~~~~~~~~~~~~~~~~~
    After each update, the feedback coefficients ``w[:M]`` are stabilized by
    reflecting any poles outside the unit circle back inside (pole reflection).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 3rd ed., Algorithm 10.3.
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
        super().__init__(filter_order=zeros_order + poles_order, w_init=w_init)

        self.zeros_order = int(zeros_order)
        self.poles_order = int(poles_order)
        self.forgetting_factor = float(forgetting_factor)
        self.epsilon = float(epsilon)

        self.n_coeffs = int(self.poles_order + self.zeros_order + 1)
        self.w = np.zeros(self.n_coeffs, dtype=np.float64)

        self.Sd = (1.0 / self.epsilon) * np.eye(self.n_coeffs, dtype=np.float64)

        self.y_buffer = np.zeros(self.poles_order, dtype=np.float64)
        self.d_buffer = np.zeros(self.poles_order, dtype=np.float64)

    def _stability_procedure(self, a_coeffs: np.ndarray) -> np.ndarray:
        """
        Enforces IIR stability by reflecting poles outside the unit circle back inside.
        This ensures the recursive part of the filter does not diverge.
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
        Executes the equation-error RLS adaptation loop.

        Parameters
        ----------
        input_signal : array_like of float
            Real-valued input sequence ``x[k]`` with shape ``(N,)``.
        desired_signal : array_like of float
            Real-valued desired/reference sequence ``d[k]`` with shape ``(N,)``.
            Must have the same length as ``input_signal``.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the time history of the feedback (pole)
            coefficients in ``result.extra["a_coefficients"]`` with shape
            ``(N, poles_order)`` (or None if ``poles_order == 0``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                "True IIR" output sequence ``y[k]`` computed with past outputs.
            - errors : ndarray of float, shape ``(N,)``
                Output error sequence ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"equation_error"``.
            - extra : dict
                Always includes:
                - ``"auxiliary_errors"``: ndarray of float, shape ``(N,)`` with
                  the equation error ``e_eq[k] = d[k] - y_eq[k]`` used to drive
                  the RLS update.
                Additionally includes ``"a_coefficients"`` if
                ``return_internal_states=True``.
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

        extra: Dict[str, Any] = {"auxiliary_errors": errors_aux}
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