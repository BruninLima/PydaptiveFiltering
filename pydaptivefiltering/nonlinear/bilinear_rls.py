#  nonlinear.bilinear_rls.py
#
#       Implements the Bilinear RLS algorithm for REAL valued data.
#       (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical
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
from typing import Optional, Union, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class BilinearRLS(AdaptiveFilter):
    """
    Bilinear RLS adaptive filter (real-valued).

    RLS algorithm with a fixed 4-dimensional *bilinear* regressor structure,
    following Diniz (Alg. 11.3). The regressor couples the current input with
    past desired samples to model a simple bilinear relationship.

    Parameters
    ----------
    forgetting_factor : float, optional
        Forgetting factor ``lambda`` with ``0 < lambda <= 1``. Default is 0.98.
    delta : float, optional
        Regularization parameter used to initialize the inverse correlation
        matrix as ``P(0) = I/delta`` (requires ``delta > 0``). Default is 1.0.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(4,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant used to guard denominators. Default is 1e-12.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Bilinear regressor (as implemented)
        This implementation uses a 4-component regressor:

        .. math::
            u[k] =
            \\begin{bmatrix}
                x[k] \\\\
                d[k-1] \\\\
                x[k]d[k-1] \\\\
                x[k-1]d[k-1]
            \\end{bmatrix}
            \\in \\mathbb{R}^{4}.

        The state ``x[k-1]`` and ``d[k-1]`` are taken from the previous iteration,
        with ``x[-1] = 0`` and ``d[-1] = 0`` at initialization.

    RLS recursion (a priori form)
        With

        .. math::
            y[k] = w^T[k-1] u[k], \\qquad e[k] = d[k] - y[k],

        the gain vector is

        .. math::
            g[k] = \\frac{P[k-1] u[k]}{\\lambda + u^T[k] P[k-1] u[k]},

        the inverse correlation update is

        .. math::
            P[k] = \\frac{1}{\\lambda}\\left(P[k-1] - g[k] u^T[k] P[k-1]\\right),

        and the coefficient update is

        .. math::
            w[k] = w[k-1] + g[k] e[k].

    Implementation details
        - The denominator ``lambda + u^T P u`` is guarded by ``safe_eps`` to avoid
          numerical issues when very small.
        - Coefficient history is recorded via the base class.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.3.
    """

    supports_complex: bool = False

    def __init__(
        self,
        forgetting_factor: float = 0.98,
        delta: float = 1.0,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        n_coeffs = 4
        super().__init__(filter_order=n_coeffs - 1, w_init=w_init)

        self.lambda_factor = float(forgetting_factor)
        if not (0.0 < self.lambda_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must satisfy 0 < forgetting_factor <= 1. Got {self.lambda_factor}."
            )

        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError(f"delta must be > 0. Got delta={self.delta}.")

        self._safe_eps = float(safe_eps)

        self.P = np.eye(n_coeffs, dtype=np.float64) / self.delta

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
        Executes the bilinear RLS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"P_last"``, ``"last_regressor"`` (``u[k]``), and ``"last_gain"`` (``g[k]``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w^T[k-1] u[k]``.
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
        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        x_prev = 0.0
        d_prev = 0.0

        last_u: Optional[np.ndarray] = None
        last_k: Optional[np.ndarray] = None

        for k in range(n_samples):
            u = np.array(
                [x[k], d_prev, x[k] * d_prev, x_prev * d_prev],
                dtype=np.float64,
            )
            last_u = u

            y_k = float(np.dot(self.w, u))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            Pu = self.P @ u
            denom = float(self.lambda_factor + (u @ Pu))
            if abs(denom) < self._safe_eps:
                denom = float(np.sign(denom) * self._safe_eps) if denom != 0.0 else float(self._safe_eps)

            k_gain = Pu / denom
            last_k = k_gain

            self.P = (self.P - np.outer(k_gain, Pu)) / self.lambda_factor

            self.w = self.w + k_gain * e_k
            self._record_history()

            x_prev = float(x[k])
            d_prev = float(d[k])

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[BilinearRLS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "P_last": self.P.copy(),
                "last_regressor": None if last_u is None else last_u.copy(),
                "last_gain": None if last_k is None else last_k.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF