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
from typing import Optional, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.validation import ensure_real_signals
from pydaptivefiltering._utils.typing import ArrayLike


class SignError(AdaptiveFilter):
    """
    Sign-Error LMS adaptive filter (real-valued).

    Low-complexity LMS variant that replaces the instantaneous error by its sign.
    This reduces multiplications and can improve robustness under impulsive noise
    in some scenarios, at the expense of slower convergence and/or larger
    steady-state misadjustment.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    At iteration ``k``, form the regressor vector (newest sample first):

    .. math::
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{R}^{M+1}.

    The a priori output and error are

    .. math::
        y[k] = w^T[k] x_k, \\qquad e[k] = d[k] - y[k].

    The sign-error update implemented here is

    .. math::
        w[k+1] = w[k] + \\mu\\, \\operatorname{sign}(e[k])\\, x_k.

    Implementation details
        - ``numpy.sign(0) = 0``; therefore if ``e[k] == 0`` the update is null.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.1 (sign-based LMS variants).
    """

    supports_complex: bool = False
    step_size: float
    def __init__(
        self,
        filter_order: int,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
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
        Executes the Sign-Error LMS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal state in ``result.extra``:
            ``"last_sign_error"`` (``sign(e[k])``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar output sequence, ``y[k] = w^T[k] x_k``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
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