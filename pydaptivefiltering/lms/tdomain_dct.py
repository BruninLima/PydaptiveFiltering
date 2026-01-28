#  lms.tdomain_dct.py
#
#       Implements the Transform-Domain LMS algorithm, based on the Discrete
#       Cosine Transform (DCT) Matrix, for COMPLEX valued data.
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
from typing import Optional, Union, Dict, Any, List

import numpy as np
from scipy.fftpack import dct

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class TDomainDCT(AdaptiveFilter):
    """
    Transform-Domain LMS using an orthonormal DCT (complex-valued).

    Transform-domain LMS algorithm (Diniz, Alg. 4.4) in which the time-domain
    regressor vector is mapped to a decorrelated transform domain using an
    orthonormal Discrete Cosine Transform (DCT). Adaptation is performed in the
    transform domain with per-bin normalization based on a smoothed power
    estimate. The time-domain coefficient vector is recovered from the
    transform-domain weights.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    gamma : float
        Regularization factor ``gamma`` used in the per-bin normalization
        denominator to avoid division by zero (or near-zero power).
    alpha : float
        Smoothing factor ``alpha`` for the transform-bin power estimate,
        typically close to 1.
    initial_power : float
        Initial power estimate used to initialize all transform bins.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of complex, optional
        Initial time-domain coefficient vector ``w(0)`` with shape ``(M + 1,)``.
        If None, initializes with zeros.

    Notes
    -----
    At iteration ``k``, form the time-domain regressor vector (newest sample first):

    .. math::
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{C}^{M+1}.

    Let ``T`` be the orthonormal DCT matrix of size ``(M+1) x (M+1)``
    (real-valued, with ``T^T T = I``). The transform-domain regressor is

    .. math::
        z_k = T x_k.

    Adaptation is performed in the transform domain with weights ``w_z[k]``.
    The a priori output and error are

    .. math::
        y[k] = w_z^H[k] z_k, \\qquad e[k] = d[k] - y[k].

    A smoothed per-bin power estimate ``p[k]`` is updated as

    .. math::
        p[k] = \\alpha\\,|z_k|^2 + (1-\\alpha)\\,p[k-1],

    where ``|z_k|^2`` is taken element-wise (i.e., ``|z_{k,i}|^2``).

    The normalized transform-domain LMS update used here is

    .. math::
        w_z[k+1] = w_z[k] + \\mu\\, e^*[k] \\, \\frac{z_k}{\\gamma + p[k]},

    where the division is element-wise.

    The time-domain coefficients are recovered using orthonormality of ``T``:

    .. math::
        w[k] = T^T w_z[k].

    Implementation details
        - ``OptimizationResult.coefficients`` stores the **time-domain** coefficient
          history recorded by the base class (``self.w`` after the inverse transform).
        - If ``return_internal_states=True``, the transform-domain coefficient history
          is returned in ``result.extra["coefficients_dct"]``.

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
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.step_size = float(step_size)

        self.N = int(self.filter_order + 1)

        self.T = dct(np.eye(self.N), norm="ortho", axis=0)

        self.w_dct = self.T @ np.asarray(self.w, dtype=complex)

        self.power_vector = np.full(self.N, float(initial_power), dtype=float)

        self._w_history_dct: List[np.ndarray] = [self.w_dct.copy()]

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Transform-Domain LMS (DCT) adaptation loop.

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
            ``"coefficients_dct"``, ``"power_vector_last"``, and ``"dct_matrix"``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w_z^H[k] z_k``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                **Time-domain** coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``coefficients_dct`` : ndarray of complex
                    Transform-domain coefficient history.
                - ``power_vector_last`` : ndarray of float
                    Final per-bin power estimate ``p[k]``.
                - ``dct_matrix`` : ndarray of float
                    The DCT matrix ``T`` used (shape ``(M+1, M+1)``).
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

        w_hist_dct: List[np.ndarray] = [self.w_dct.copy()]

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]
            z_k = self.T @ x_k

            self.power_vector = (
                self.alpha * np.real(z_k * np.conj(z_k)) + (1.0 - self.alpha) * self.power_vector
            )

            y_k = complex(np.vdot(self.w_dct, z_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            denom = self.gamma + self.power_vector
            self.w_dct = self.w_dct + self.step_size * np.conj(e_k) * (z_k / denom)

            self.w = self.T.T @ self.w_dct

            self._record_history()
            w_hist_dct.append(self.w_dct.copy())

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[TDomainDCT] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "coefficients_dct": np.asarray(w_hist_dct),
                "power_vector_last": self.power_vector.copy(),
                "dct_matrix": self.T.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF