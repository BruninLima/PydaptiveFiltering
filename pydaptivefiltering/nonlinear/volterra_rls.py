#  nonlinear.volterra_rls.py
#
#       Implements the Volterra RLS algorithm for REAL valued data.
#       (Algorithm 11.2 - book: Adaptive Filtering: Algorithms and Practical
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

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class VolterraRLS(AdaptiveFilter):
    """
    Second-order Volterra RLS adaptive filter (real-valued).

    Volterra RLS (Diniz, Alg. 11.2) using a second-order Volterra expansion and
    an RLS update applied to the expanded regressor. The model augments a linear
    tapped-delay regressor with all unique quadratic products (including
    squares) and estimates the corresponding coefficient vector via RLS.

    Parameters
    ----------
    memory : int, optional
        Linear memory length ``L``. The linear delay line is
        ``[x[k], x[k-1], ..., x[k-L+1]]``. Default is 3.
    forgetting_factor : float, optional
        Forgetting factor ``lambda`` with ``0 < lambda <= 1``. Default is 0.98.
    delta : float, optional
        Regularization parameter used to initialize the inverse correlation
        matrix as ``P(0) = I/delta`` (requires ``delta > 0``). Default is 1.0.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(n_coeffs,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant used to guard denominators. Default is 1e-12.

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
        for ``0 \\le i \\le j \\le L-1``.

        The number of coefficients is

        .. math::
            n_{coeffs} = L + \\frac{L(L+1)}{2}.

    RLS recursion (a priori form)
        With

        .. math::
            y[k] = w^T[k-1] u[k], \\qquad e[k] = d[k] - y[k],

        define the gain

        .. math::
            g[k] = \\frac{P[k-1] u[k]}{\\lambda + u^T[k] P[k-1] u[k]},

        the inverse correlation update

        .. math::
            P[k] = \\frac{1}{\\lambda}\\left(P[k-1] - g[k] u^T[k] P[k-1]\\right),

        and the coefficient update

        .. math::
            w[k] = w[k-1] + g[k] e[k].

    A posteriori quantities
        If requested, this implementation also computes the *a posteriori*
        output/error after updating the coefficients at time ``k``:

        .. math::
            y^{post}[k] = w^T[k] u[k], \\qquad e^{post}[k] = d[k] - y^{post}[k].

    Implementation details
        - The denominator ``lambda + u^T P u`` is guarded by ``safe_eps`` to avoid
          numerical issues when very small.
        - Coefficient history is recorded via the base class.
        - The quadratic-term ordering matches :meth:`_create_volterra_regressor`.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.2.
    """

    supports_complex: bool = False

    def __init__(
        self,
        memory: int = 3,
        forgetting_factor: float = 0.98,
        delta: float = 1.0,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        memory:
            Linear memory length L. Determines number of Volterra coefficients:
            n_coeffs = L + L(L+1)/2.
        forgetting_factor:
            Forgetting factor λ (typically close to 1). Must satisfy 0 < λ <= 1.
        delta:
            Positive regularization for initializing the inverse correlation matrix:
            P[0] = I / delta.
        w_init:
            Optional initial coefficient vector (length n_coeffs). If None, zeros.
        safe_eps:
            Small epsilon to guard denominators.
        """
        memory = int(memory)
        if memory <= 0:
            raise ValueError(f"memory must be > 0. Got {memory}.")

        lam = float(forgetting_factor)
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"forgetting_factor must satisfy 0 < λ <= 1. Got λ={lam}.")

        delta = float(delta)
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0. Got delta={delta}.")

        self.memory: int = memory
        self.lam: float = lam
        self._safe_eps: float = float(safe_eps)

        self.n_coeffs: int = memory + (memory * (memory + 1)) // 2

        super().__init__(filter_order=self.n_coeffs - 1, w_init=w_init)

        self.w = np.asarray(self.w, dtype=np.float64)

        if w_init is not None:
            w0 = np.asarray(w_init, dtype=np.float64).reshape(-1)
            if w0.size != self.n_coeffs:
                raise ValueError(
                    f"w_init must have length {self.n_coeffs}, got {w0.size}."
                )
            self.w = w0.copy()

        self.P: np.ndarray = (np.eye(self.n_coeffs, dtype=np.float64) / delta)

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
            raise ValueError(f"x_lin must have length {self.memory}, got {x_lin.size}.")

        quad = np.empty((self.memory * (self.memory + 1)) // 2, dtype=np.float64)
        idx = 0
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad[idx] = x_lin[i] * x_lin[j]
                idx += 1

        return np.concatenate([x_lin, quad], axis=0)

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
        Executes the Volterra RLS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes additional internal sequences in ``result.extra``,
            including a posteriori output/error and last gain/denominator.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w^T[k-1] u[k]``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Volterra coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``posteriori_outputs`` : ndarray of float
                    A posteriori output sequence ``y^{post}[k]``.
                - ``posteriori_errors`` : ndarray of float
                    A posteriori error sequence ``e^{post}[k]``.
                - ``last_gain`` : ndarray of float
                    Last RLS gain vector ``g[k]``.
                - ``last_den`` : float
                    Last gain denominator ``lambda + u^T P u``.
                - ``last_regressor`` : ndarray of float
                    Last Volterra regressor ``u[k]``.
                - ``memory`` : int
                    Linear memory length ``L``.
                - ``n_coeffs`` : int
                    Number of Volterra coefficients.
                - ``forgetting_factor`` : float
                    The forgetting factor ``lambda`` used.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(x.size)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        y_post = np.zeros(n_samples, dtype=np.float64)
        e_post = np.zeros(n_samples, dtype=np.float64)

        L = int(self.memory)
        x_padded = np.zeros(n_samples + (L - 1), dtype=np.float64)
        x_padded[L - 1 :] = x

        last_k: Optional[np.ndarray] = None
        last_den: Optional[float] = None
        last_u: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_lin = x_padded[k : k + L][::-1]
            u = self._create_volterra_regressor(x_lin)
            last_u = u

            y_k = float(np.dot(self.w, u))
            e_k = float(d[k] - y_k)
            outputs[k] = y_k
            errors[k] = e_k

            Pu = self.P @ u
            den = float(self.lam + np.dot(u, Pu))
            if abs(den) < self._safe_eps:
                den = float(den + np.sign(den) * self._safe_eps) if den != 0.0 else float(self._safe_eps)

            k_gain = Pu / den
            last_k = k_gain
            last_den = den

            self.w = self.w + k_gain * e_k

            self.P = (self.P - np.outer(k_gain, Pu)) / self.lam

            yk_post = float(np.dot(self.w, u))
            ek_post = float(d[k] - yk_post)
            y_post[k] = yk_post
            e_post[k] = ek_post

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[VolterraRLS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "posteriori_outputs": y_post,
                "posteriori_errors": e_post,
                "last_gain": None if last_k is None else last_k.copy(),
                "last_den": last_den,
                "last_regressor": None if last_u is None else last_u.copy(),
                "memory": int(self.memory),
                "n_coeffs": int(self.n_coeffs),
                "forgetting_factor": float(self.lam),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF