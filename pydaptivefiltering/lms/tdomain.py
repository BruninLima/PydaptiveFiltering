#  lms.tdomain.py
#
#       Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
#       (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
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
from typing import Optional, Dict, Any, List

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.typing import ArrayLike


class TDomainLMS(AdaptiveFilter):
    """
    Transform-Domain LMS with a user-provided transform matrix.

    Generic transform-domain LMS algorithm (Diniz, Alg. 4.4) parameterized by a
    transform matrix ``T``. At each iteration, the time-domain regressor is
    mapped to the transform domain, adaptation is performed with per-bin
    normalization using a smoothed power estimate, and time-domain coefficients
    are recovered from the transform-domain weights.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
        The transform size must be ``(M + 1, M + 1)``.
    gamma : float
        Regularization factor ``gamma`` used in the per-bin normalization
        denominator to avoid division by zero (or near-zero power).
    alpha : float
        Smoothing factor ``alpha`` for the transform-bin power estimate,
        typically close to 1.
    initial_power : float
        Initial power estimate used to initialize all transform bins.
    transform_matrix : array_like of complex
        Transform matrix ``T`` with shape ``(M + 1, M + 1)``.
        Typically unitary (``T^H T = I``).
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of complex, optional
        Initial **time-domain** coefficient vector ``w(0)`` with shape ``(M + 1,)``.
        If None, initializes with zeros.
    assume_unitary : bool, optional
        If True (default), maps transform-domain weights back to the time domain
        using ``w = T^H w_T`` (fast). If False, uses a pseudo-inverse mapping
        ``w = pinv(T)^H w_T`` (slower but works for non-unitary ``T``).

    Notes
    -----
    At iteration ``k``, form the time-domain regressor vector (newest sample first):

    .. math::
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{C}^{M+1}.

    Define the transform-domain regressor:

    .. math::
        z_k = T x_k.

    Adaptation is performed in the transform domain with weights ``w_T[k]``.
    The a priori output and error are

    .. math::
        y[k] = w_T^H[k] z_k, \\qquad e[k] = d[k] - y[k].

    A smoothed per-bin power estimate ``p[k]`` is updated as

    .. math::
        p[k] = \\alpha\\,|z_k|^2 + (1-\\alpha)\\,p[k-1],

    where ``|z_k|^2`` is taken element-wise.

    The normalized transform-domain LMS update used here is

    .. math::
        w_T[k+1] = w_T[k] + \\mu\\, e^*[k] \\, \\frac{z_k}{\\gamma + p[k]},

    with element-wise division.

    Mapping back to time domain
        If ``T`` is unitary (``T^H T = I``), then the inverse mapping is

        .. math::
            w[k] = T^H w_T[k].

        If ``T`` is not unitary and ``assume_unitary=False``, this implementation
        uses the pseudo-inverse mapping:

        .. math::
            w[k] = \\operatorname{pinv}(T)^H w_T[k].

    Implementation details
        - ``OptimizationResult.coefficients`` stores the **time-domain** coefficient
          history recorded by the base class (``self.w`` after mapping back).
        - If ``return_internal_states=True``, the transform-domain coefficient history
          is returned in ``result.extra["coefficients_transform"]``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.4.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        gamma: float,
        alpha: float,
        initial_power: float,
        transform_matrix: np.ndarray,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        assume_unitary: bool = True,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.step_size = float(step_size)

        self.N = int(self.filter_order + 1)

        T = np.asarray(transform_matrix, dtype=complex)
        if T.shape != (self.N, self.N):
            raise ValueError(f"transform_matrix must have shape {(self.N, self.N)}. Got {T.shape}.")

        self.T = T
        self._assume_unitary = bool(assume_unitary)

        # transform-domain weights (start from time-domain w)
        self.w_T = self.T @ np.asarray(self.w, dtype=complex)

        # power estimate per transform bin
        self.power_vector = np.full(self.N, float(initial_power), dtype=float)

        # optional transform-domain history
        self._w_history_T: List[np.ndarray] = [self.w_T.copy()]

    def _to_time_domain(self, w_T: np.ndarray) -> np.ndarray:
        """Map transform-domain weights to time-domain weights."""
        if self._assume_unitary:
            return self.T.conj().T @ w_T
        # fallback for non-unitary transforms (more expensive)
        T_pinv = np.linalg.pinv(self.T)
        return T_pinv.conj().T @ w_T

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Transform-Domain LMS adaptation loop.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes transform-domain internal states in ``result.extra``:
            ``"coefficients_transform"``, ``"power_vector_last"``,
            ``"transform_matrix"``, and ``"assume_unitary"``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w_T^H[k] z_k``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                **Time-domain** coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``coefficients_transform`` : ndarray of complex
                    Transform-domain coefficient history.
                - ``power_vector_last`` : ndarray of float
                    Final per-bin power estimate ``p[k]``.
                - ``transform_matrix`` : ndarray of complex
                    The transform matrix ``T`` used (shape ``(M+1, M+1)``).
                - ``assume_unitary`` : bool
                    Whether the inverse mapping assumed ``T`` is unitary.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=complex).ravel()
        d = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        x_padded = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        w_hist_T: List[np.ndarray] = [self.w_T.copy()]

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]
            z_k = self.T @ x_k

            self.power_vector = (
                self.alpha * np.real(z_k * np.conj(z_k)) + (1.0 - self.alpha) * self.power_vector
            )

            y_k = complex(np.vdot(self.w_T, z_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            denom = self.gamma + self.power_vector
            self.w_T = self.w_T + self.step_size * np.conj(e_k) * (z_k / denom)

            self.w = self._to_time_domain(self.w_T)

            self._record_history()
            w_hist_T.append(self.w_T.copy())

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[TDomainLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "coefficients_transform": np.asarray(w_hist_T),
                "power_vector_last": self.power_vector.copy(),
                "transform_matrix": self.T.copy(),
                "assume_unitary": self._assume_unitary,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF