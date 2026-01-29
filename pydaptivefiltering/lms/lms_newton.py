#  lms.lms_newton.py
#
#       Implements the Complex LMS-Newton algorithm for COMPLEX valued data.
#       (Algorithm 4.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
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
from typing import Optional

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.typing import ArrayLike


class LMSNewton(AdaptiveFilter):
    """
    Complex LMS-Newton adaptive filter.

    LMS-Newton accelerates the standard complex LMS by preconditioning the
    instantaneous gradient with a recursive estimate of the inverse input
    correlation matrix. This often improves convergence speed for strongly
    correlated inputs, at the cost of maintaining and updating a full
    ``(M+1) x (M+1)`` matrix per iteration.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    forgetting_factor : float
        Forgetting factor ``alpha`` used in the inverse-correlation recursion,
        with ``0 < forgetting_factor < 1``. Values closer to 1 yield smoother tracking; smaller
        values adapt faster.
    initial_inv_rx : array_like of complex
        Initial inverse correlation matrix ``P(0)`` with shape ``(M + 1, M + 1)``.
        Typical choices are scaled identities, e.g. ``delta^{-1} I``.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant used to guard denominators in the matrix recursion.
        Default is 1e-12.

    Notes
    -----
    Complex-valued
        This implementation assumes complex arithmetic (``supports_complex=True``),
        with the a priori output computed as ``y[k] = w^H[k] x_k``.

    Recursion (as implemented)
        Let the regressor vector be

        .. math::
            x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{C}^{M+1},

        and define the output and a priori error as

        .. math::
            y[k] = w^H[k] x_k, \\qquad e[k] = d[k] - y[k].

        Maintain an estimate ``P[k] \\approx R_x^{-1}`` using a normalized rank-1 update.
        With

        .. math::
            p_k = P[k] x_k, \\qquad \\phi_k = x_k^H p_k,

        the denominator is

        .. math::
            \\mathrm{denom}_k = \\frac{1-\\text{forgetting_factor}}{\\text{forgetting_factor}} + \\phi_k,

        and the update used here is

        .. math::
            P[k+1] =
            \\frac{1}{1-\\text{forgetting_factor}}
            \\left(
                P[k] - \\frac{p_k p_k^H}{\\mathrm{denom}_k}
            \\right).

        The coefficient update uses the preconditioned regressor ``P[k+1] x_k``:

        .. math::
            w[k+1] = w[k] + \\mu\\, e^*[k] \\, (P[k+1] x_k).

    Relationship to RLS
        The recursion for ``P`` is algebraically similar to an RLS covariance update
        with a particular normalization; however, the coefficient update remains
        LMS-like, controlled by the step size ``mu``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.2.
    """

    supports_complex: bool = True

    forgetting_factor: float
    step_size: float
    inv_rx: np.ndarray

    def __init__(
        self,
        filter_order: int,
        forgetting_factor: float,
        initial_inv_rx: np.ndarray,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.forgetting_factor = float(forgetting_factor)
        if not (0.0 < self.forgetting_factor < 1.0):
            raise ValueError(f"forgetting_factor must satisfy 0 < forgetting_factor < 1. Got forgetting_factor={self.forgetting_factor}.")

        P0 = np.asarray(initial_inv_rx, dtype=complex)
        n_taps = int(filter_order) + 1
        if P0.shape != (n_taps, n_taps):
            raise ValueError(
                f"initial_inv_rx must have shape {(n_taps, n_taps)}. Got {P0.shape}."
            )
        self.inv_rx = P0

        self.step_size = float(step_size)
        self._safe_eps = float(safe_eps)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Executes the LMS-Newton adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar output sequence, ``y[k] = w^H[k] x_k``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)
        m: int = int(self.filter_order)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + m + 1][::-1]

            y_k: complex = complex(np.vdot(self.w, x_k))
            outputs[k] = y_k

            e_k: complex = d[k] - y_k
            errors[k] = e_k

            x_col: np.ndarray = x_k.reshape(-1, 1)
            Px: np.ndarray = self.inv_rx @ x_col
            phi: complex = (x_col.conj().T @ Px).item()

            denom: complex = ((1.0 - self.forgetting_factor) / self.forgetting_factor) + phi
            if abs(denom) < self._safe_eps:
                denom = denom + (self._safe_eps + 0.0j)

            self.inv_rx = (self.inv_rx - (Px @ Px.conj().T) / denom) / (1.0 - self.forgetting_factor)

            self.w = self.w + self.step_size * np.conj(e_k) * Px.ravel()

            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[LMSNewton] Completed in {runtime_s * 1000:.03f} ms")

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
        )
# EOF