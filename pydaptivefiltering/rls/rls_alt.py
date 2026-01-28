#  rls.rls_alt.py
#
#       Implements the Alternative RLS algorithm for COMPLEX valued data.
#       RLS_Alt differs from RLS in the number of computations. The RLS_Alt
#       uses an auxiliary variable (psi) in order to reduce the computational burden.
#       (Algorithm 5.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import time
from typing import Any, Dict, Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class RLSAlt(AdaptiveFilter):
    """
    Alternative RLS (RLS-Alt) adaptive filter (complex-valued).

    Alternative RLS algorithm based on Diniz (Alg. 5.4), designed to reduce
    the computational burden of the standard RLS recursion by introducing an
    auxiliary vector ``psi[k]``. The method maintains an estimate of the inverse
    input correlation matrix and updates the coefficients using a Kalman-gain-like
    vector.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    delta : float
        Positive initialization factor for the inverse correlation matrix:
        ``S_d(0) = (1/delta) I``.
    forgetting_factor : float
        Forgetting factor ``lambda`` with ``0 < lambda <= 1``.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.
    safe_eps : float, optional
        Small positive constant used to guard denominators. Default is 1e-12.

    Notes
    -----
    At iteration ``k``, form the regressor vector (tapped delay line):

    - ``x_k = [x[k], x[k-1], ..., x[k-M]]^T  ∈ 𝕮^{M+1}``

    The a priori output and error are:

    .. math::
        y[k] = w^H[k] x_k, \\qquad e[k] = d[k] - y[k].

    The key auxiliary vector is:

    .. math::
        \\psi[k] = S_d[k-1] x_k,

    where ``S_d[k-1]`` is the inverse correlation estimate.

    Define the gain denominator:

    .. math::
        \\Delta[k] = \\lambda + x_k^H \\psi[k]
                   = \\lambda + x_k^H S_d[k-1] x_k,

    and the gain vector:

    .. math::
        g[k] = \\frac{\\psi[k]}{\\Delta[k]}.

    The coefficient update is:

    .. math::
        w[k+1] = w[k] + e^*[k] \\, g[k],

    and the inverse correlation update is:

    .. math::
        S_d[k] = \\frac{1}{\\lambda}\\Bigl(S_d[k-1] - g[k] \\psi^H[k]\\Bigr).

    A posteriori quantities
        If requested, this implementation also computes the *a posteriori*
        output/error using the updated weights:

        .. math::
            y^{post}[k] = w^H[k+1] x_k, \\qquad e^{post}[k] = d[k] - y^{post}[k].

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 5.4.
    """

    supports_complex: bool = True

    forgetting_factor: float
    delta: float
    S_d: np.ndarray

    def __init__(
        self,
        filter_order: int,
        delta: float,
        forgetting_factor: float,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.forgetting_factor = float(forgetting_factor)
        if not (0.0 < self.forgetting_factor <= 1.0):
            raise ValueError(f"forgetting_factor must satisfy 0 < forgetting_factor <= 1. Got forgetting_factor={self.forgetting_factor}.")

        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError(f"delta must be positive. Got delta={self.delta}.")

        self._safe_eps = float(safe_eps)

        n_taps = int(self.filter_order) + 1
        self.S_d = (1.0 / self.delta) * np.eye(n_taps, dtype=complex)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the RLS-Alt adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes a posteriori sequences and the last internal states
            in ``result.extra`` (see below).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                A priori output sequence, ``y[k] = w^H[k] x_k``.
            - errors : ndarray of complex, shape ``(N,)``
                A priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``outputs_posteriori`` : ndarray of complex
                    A posteriori output sequence, ``y^{post}[k] = w^H[k+1] x_k``.
                - ``errors_posteriori`` : ndarray of complex
                    A posteriori error sequence, ``e^{post}[k] = d[k] - y^{post}[k]``.
                - ``S_d_last`` : ndarray of complex
                    Final inverse correlation matrix ``S_d``.
                - ``gain_last`` : ndarray of complex
                    Last gain vector ``g[k]``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        y_post: Optional[np.ndarray] = None
        e_post: Optional[np.ndarray] = None
        if return_internal_states:
            y_post = np.zeros(n_samples, dtype=complex)
            e_post = np.zeros(n_samples, dtype=complex)

        last_gain: Optional[np.ndarray] = None

        for k in range(n_samples):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x[k]

            y_k = complex(np.vdot(self.w, self.regressor))
            e_k = d[k] - y_k

            outputs[k] = y_k
            errors[k] = e_k

            psi: np.ndarray = self.S_d @ self.regressor

            den: complex = self.forgetting_factor + complex(np.vdot(self.regressor, psi))
            if abs(den) < self._safe_eps:
                den = den + (self._safe_eps + 0.0j)

            g: np.ndarray = psi / den
            last_gain = g

            self.w = self.w + np.conj(e_k) * g

            self.S_d = (self.S_d - np.outer(g, np.conj(psi))) / self.forgetting_factor

            if return_internal_states:
                yk_post = complex(np.vdot(self.w, self.regressor))
                y_post[k] = yk_post
                e_post[k] = d[k] - yk_post

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[RLSAlt] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "outputs_posteriori": y_post,
                "errors_posteriori": e_post,
                "S_d_last": self.S_d.copy(),
                "gain_last": None if last_gain is None else last_gain.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF