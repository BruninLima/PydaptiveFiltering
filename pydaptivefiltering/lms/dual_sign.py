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
from typing import Optional

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.validation import ensure_real_signals
from pydaptivefiltering._utils.typing import ArrayLike


class DualSign(AdaptiveFilter):
    """
    Dual-Sign LMS (DS-LMS) adaptive filter (real-valued).

    Low-complexity LMS variant that uses the *sign* of the instantaneous error
    and a two-level (piecewise) effective gain selected by the error magnitude.
    This can reduce the number of multiplications and may improve robustness
    under impulsive noise in some scenarios, at the expense of larger steady-state
    misadjustment.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    rho : float
        Threshold ``rho`` applied to ``|e[k]|`` to select the gain level.
    gamma : float
        Gain multiplier applied when ``|e[k]| > rho`` (typically ``gamma > 1``).
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant kept for API consistency across the library.
        (Not used by this implementation.) Default is 1e-12.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Update rule (as implemented)
        Let the regressor vector be

        .. math::
            x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T,

        with output and error

        .. math::
            y[k] = w^T[k] x_k, \\qquad e[k] = d[k] - y[k].

        Define the two-level signed term

        .. math::
            u[k] =
            \\begin{cases}
                \\operatorname{sign}(e[k]), & |e[k]| \\le \\rho \\\\
                \\gamma\\,\\operatorname{sign}(e[k]), & |e[k]| > \\rho
            \\end{cases}

        and update

        .. math::
            w[k+1] = w[k] + 2\\mu\\,u[k]\,x_k.

    Implementation details
        - ``numpy.sign(0) = 0``; therefore if ``e[k] == 0`` the update is null.
        - The factor ``2`` in the update matches the implementation in this
          module (consistent with common LMS gradient conventions).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.1 (modified sign-based variant).
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
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.step_size = float(step_size)
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
        Executes the DS-LMS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.

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