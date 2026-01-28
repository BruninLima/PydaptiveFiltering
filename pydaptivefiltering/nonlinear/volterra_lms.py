#  nonlinear.volterra_lms.py
#
#       Implements the Volterra LMS algorithm for REAL valued data.
#       (Algorithm 11.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult
from pydaptivefiltering._utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class VolterraLMS(AdaptiveFilter):
    """
    Second-order Volterra LMS adaptive filter (real-valued).

    Volterra LMS (Diniz, Alg. 11.1) using a second-order Volterra expansion.
    The adaptive model augments a linear tapped-delay regressor with all
    quadratic products (including squares) and performs an LMS-type update on
    the expanded coefficient vector.

    Parameters
    ----------
    memory : int, optional
        Linear memory length ``L``. The linear delay line is
        ``[x[k], x[k-1], ..., x[k-L+1]]``. Default is 3.
    step_size : float or array_like of float, optional
        Step size ``mu``. Can be either:
        - a scalar (same step for all coefficients), or
        - a vector with shape ``(n_coeffs,)`` for per-term step scaling.
        Default is 1e-2.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(n_coeffs,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant kept for API consistency across the library.
        (Not used directly by this implementation.) Default is 1e-12.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Volterra regressor (as implemented)
        Let the linear delay line be

        .. math::
            x_{lin}[k] = [x[k], x[k-1], \\ldots, x[k-L+1]]^T \\in \\mathbb{R}^{L}.

        The second-order Volterra regressor is constructed as

        .. math::
            u[k] =
            \\begin{bmatrix}
                x_{lin}[k] \\\\
                \\mathrm{vec}\\bigl(x_{lin}[k] x_{lin}^T[k]\\bigr)_{i \\le j}
            \\end{bmatrix}
            \\in \\mathbb{R}^{n_{coeffs}},

        where the quadratic block contains all products ``x_{lin,i}[k] x_{lin,j}[k]``
        for ``0 \\le i \\le j \\le L-1`` (unique terms only).

        The number of coefficients is therefore

        .. math::
            n_{coeffs} = L + \\frac{L(L+1)}{2}.

    LMS recursion (a priori)
        With

        .. math::
            y[k] = w^T[k] u[k], \\qquad e[k] = d[k] - y[k],

        the update implemented here is

        .. math::
            w[k+1] = w[k] + 2\\mu\\, e[k] \\, u[k],

        where ``\\mu`` may be scalar or element-wise (vector step).

    Implementation details
        - The coefficient vector ``self.w`` stores the full Volterra parameter
          vector (linear + quadratic) and is recorded by the base class.
        - The quadratic term ordering matches the nested loops used in
          :meth:`_create_volterra_regressor` (i increasing, j from i to L-1).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.1.
    """

    supports_complex: bool = False

    def __init__(
        self,
        memory: int = 3,
        step_size: Union[float, np.ndarray, list] = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        memory = int(memory)
        if memory <= 0:
            raise ValueError(f"memory must be > 0. Got {memory}.")

        self.memory: int = memory
        self.n_coeffs: int = memory + (memory * (memory + 1)) // 2
        self._safe_eps: float = float(safe_eps)

        super().__init__(filter_order=self.n_coeffs - 1, w_init=w_init)

        if isinstance(step_size, (list, np.ndarray)):
            step_vec = np.asarray(step_size, dtype=np.float64).reshape(-1)
            if step_vec.size != self.n_coeffs:
                raise ValueError(
                    f"step vector must have length {self.n_coeffs}, got {step_vec.size}."
                )
            self.step_size: Union[float, np.ndarray] = step_vec
        else:
            self.step_size = float(step_size)

        self.w = np.asarray(self.w, dtype=np.float64)

        self.w_history = []
        self._record_history()

    def _create_volterra_regressor(self, x_lin: np.ndarray) -> np.ndarray:
        """
        Constructs the second-order Volterra regressor from a linear delay line.

        Parameters
        ----------
        x_lin : ndarray of float
            Linear delay line with shape ``(L,)`` ordered as
            ``[x[k], x[k-1], ..., x[k-L+1]]``.

        Returns
        -------
        ndarray of float
            Volterra regressor ``u[k]`` with shape ``(n_coeffs,)`` containing:
            - linear terms, followed by
            - quadratic terms for ``i <= j``.
        """
        x_lin = np.asarray(x_lin, dtype=np.float64).reshape(-1)
        if x_lin.size != self.memory:
            raise ValueError(
                f"x_lin must have length {self.memory}, got {x_lin.size}."
            )

        quad = np.empty((self.memory * (self.memory + 1)) // 2, dtype=np.float64)
        idx = 0
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad[idx] = x_lin[i] * x_lin[j]
                idx += 1

        return np.concatenate([x_lin, quad], axis=0)

    @ensure_real_signals
    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Volterra LMS adaptation loop over paired input/desired sequences.

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
            ``"last_regressor"``, ``"memory"``, and ``"n_coeffs"``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w^T[k] u[k]``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Volterra coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        if x.size != d.size:
            raise ValueError(f"Inconsistent lengths: input({x.size}) != desired({d.size})")
        n_samples = int(x.size)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        L = int(self.memory)
        x_padded = np.zeros(n_samples + (L - 1), dtype=np.float64)
        x_padded[L - 1 :] = x

        last_u: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_lin = x_padded[k : k + L][::-1]
            u = self._create_volterra_regressor(x_lin)
            last_u = u

            y_k = float(np.dot(self.w, u))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            if isinstance(self.step_size, np.ndarray):
                self.w = self.w + (2.0 * self.step_size) * e_k * u
            else:
                self.w = self.w + (2.0 * float(self.step_size)) * e_k * u

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[VolterraLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "last_regressor": None if last_u is None else last_u.copy(),
                "memory": int(self.memory),
                "n_coeffs": int(self.n_coeffs),
            }

        return self._pack_results(
            outputs=outputs,
            errors= errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF