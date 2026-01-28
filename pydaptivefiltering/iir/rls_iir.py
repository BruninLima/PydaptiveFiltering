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
from typing import Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.validation import ensure_real_signals


class RLSIIR(AdaptiveFilter):
    """
    RLS-like output-error adaptation for IIR filters (real-valued).

    This algorithm applies an RLS-style recursion to the IIR output-error (OE)
    problem. Rather than minimizing a linear FIR error, it uses filtered
    sensitivity signals to build a Jacobian-like vector :math:`\\phi(k)` that
    approximates how the IIR output changes with respect to the pole/zero
    parameters. The inverse correlation matrix (named ``Sd``) scales the update,
    typically yielding faster convergence than plain gradient methods.

    The implementation corresponds to a modified form of Diniz (3rd ed.,
    Alg. 10.1) and is restricted to **real-valued** signals (enforced by
    ``ensure_real_signals``).

    Parameters
    ----------
    zeros_order : int
        Numerator order ``N`` (number of zeros). The feedforward part has
        ``N + 1`` coefficients.
    poles_order : int
        Denominator order ``M`` (number of poles). The feedback part has ``M``
        coefficients.
    forgetting_factor : float, optional
        Exponential forgetting factor ``lambda`` used in the recursive update of
        ``Sd``. Typical values are in ``[0.9, 1.0]``. Default is 0.99.
    delta : float, optional
        Positive regularization parameter for initializing ``Sd`` as
        :math:`S(0) = \\delta^{-1} I`. Default is 1e-3.
    w_init : array_like of float, optional
        Optional initial coefficient vector. If provided, it should have shape
        ``(M + N + 1,)`` following the parameter order described below. If None,
        the implementation initializes with zeros (and ignores ``w_init``).

    Notes
    -----
    Parameterization (as implemented)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The coefficient vector is arranged as:

    - ``w[:M]``: feedback (pole) coefficients (often denoted ``a``)
    - ``w[M:]``: feedforward (zero) coefficients (often denoted ``b``)

    OE output and error (as implemented)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    With ``reg_x = [x(k), x(k-1), ..., x(k-N)]^T`` and an internal buffer of the
    last ``M`` outputs, the code forms:

    .. math::
        \\varphi(k) = [y(k-1), \\ldots, y(k-M),\\; x(k), \\ldots, x(k-N)]^T,

    computes:

    .. math::
        y(k) = w^T(k)\\, \\varphi(k), \\qquad e(k) = d(k) - y(k),

    and reports ``e(k)`` as the output-error sequence.

    Sensitivity vector and RLS recursion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Filtered sensitivity signals stored in internal buffers (``x_line_buffer``
    and ``y_line_buffer``) are used to build:

    .. math::
        \\phi(k) =
        [\\underline{y}(k-1), \\ldots, \\underline{y}(k-M),\\;
         -\\underline{x}(k), \\ldots, -\\underline{x}(k-N)]^T.

    The inverse correlation matrix ``Sd`` is updated in an RLS-like manner:

    .. math::
        \\psi(k) = Sd(k-1)\\, \\phi(k), \\quad
        \\text{den}(k) = \\lambda + \\phi^T(k)\\, \\psi(k),

    .. math::
        Sd(k) = \\frac{1}{\\lambda}
                \\left(Sd(k-1) - \\frac{\\psi(k)\\psi^T(k)}{\\text{den}(k)}\\right).

    The coefficient update used here is:

    .. math::
        w(k+1) = w(k) - Sd(k)\\, \\phi(k)\\, e(k).

    (Note: this implementation does not expose an additional step-size parameter;
    the effective step is governed by ``Sd``.)

    Stability procedure
    ~~~~~~~~~~~~~~~~~~~
    After each update, the feedback coefficients ``w[:M]`` are stabilized by
    reflecting poles outside the unit circle back inside (pole reflection).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 3rd ed., Algorithm 10.1 (modified).
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
        Executes the RLS-IIR (OE) adaptation loop.

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
            If True, includes sensitivity trajectories in ``result.extra``:
            - ``"x_sensitivity"``: ndarray of float, shape ``(N,)`` with the
              scalar sensitivity signal :math:`\\underline{x}(k)`.
            - ``"y_sensitivity"``: ndarray of float, shape ``(N,)`` with the
              scalar sensitivity signal :math:`\\underline{y}(k)`.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Output sequence ``y[k]`` produced by the current IIR structure.
            - errors : ndarray of float, shape ``(N,)``
                Output error sequence ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"output_error"``.
            - extra : dict
                Empty unless ``return_internal_states=True``.
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

            if return_internal_states and x_line_track is not None:
                x_line_track[k], y_line_track[k] = x_line_k, y_line_k

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

        extra = {"x_sensitivity": x_line_track, "y_sensitivity": y_line_track} if return_internal_states else {}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF