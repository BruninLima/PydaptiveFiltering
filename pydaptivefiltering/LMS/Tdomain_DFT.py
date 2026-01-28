#  lms.tdomain_dft.py
#
#       Implements the Transform-Domain LMS algorithm, based on the Discrete
#       Fourier Transform (DFT), for COMPLEX valued data.
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
from scipy.fft import fft, ifft

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class TDomainDFT(AdaptiveFilter):
    """
    Transform-Domain LMS using a unitary DFT (complex-valued).

    Transform-domain LMS algorithm (Diniz, Alg. 4.4) in which the time-domain
    regressor is mapped to the frequency domain using a *unitary* Discrete
    Fourier Transform (DFT). Adaptation is performed in the transform domain
    with per-bin normalization based on a smoothed power estimate. The time-domain
    coefficient vector is recovered via the inverse unitary DFT.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
        The DFT size is ``N = M + 1``.
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
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{C}^{N}.

    Define the *unitary* DFT (energy-preserving) transform-domain regressor:

    .. math::
        z_k = \\frac{\\mathrm{DFT}(x_k)}{\\sqrt{N}}.

    Adaptation is performed in the transform domain with weights ``w_z[k]``.
    The a priori output and error are

    .. math::
        y[k] = w_z^H[k] z_k, \\qquad e[k] = d[k] - y[k].

    A smoothed per-bin power estimate ``p[k]`` is updated as

    .. math::
        p[k] = \\alpha\\,|z_k|^2 + (1-\\alpha)\\,p[k-1],

    where ``|z_k|^2`` is taken element-wise.

    The normalized transform-domain LMS update used here is

    .. math::
        w_z[k+1] = w_z[k] + \\mu\\, e^*[k] \\, \\frac{z_k}{\\gamma + p[k]},

    with element-wise division.

    The time-domain coefficients are recovered via the inverse unitary DFT:

    .. math::
        w[k] = \\mathrm{IDFT}(w_z[k])\\,\\sqrt{N}.

    Implementation details
        - ``OptimizationResult.coefficients`` stores the **time-domain** coefficient
          history recorded by the base class (``self.w`` after inverse transform).
        - If ``return_internal_states=True``, the transform-domain coefficient history
          is returned in ``result.extra["coefficients_dft"]``.

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
        self._sqrtN = float(np.sqrt(self.N))

        self.w_dft = fft(np.asarray(self.w, dtype=complex)) / self._sqrtN

        self.power_vector = np.full(self.N, float(initial_power), dtype=float)

        self._w_history_dft: List[np.ndarray] = [self.w_dft.copy()]

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Transform-Domain LMS (DFT) adaptation loop.

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
            ``"coefficients_dft"``, ``"power_vector_last"``, and ``"sqrtN"``.

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
                - ``coefficients_dft`` : ndarray of complex
                    Transform-domain coefficient history.
                - ``power_vector_last`` : ndarray of float
                    Final per-bin power estimate ``p[k]``.
                - ``sqrtN`` : float
                    The unitary normalization factor ``\\sqrt{N}``.
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

        w_hist_dft: List[np.ndarray] = [self.w_dft.copy()]

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]
            z_k = fft(x_k) / self._sqrtN

            self.power_vector = (
                self.alpha * np.real(z_k * np.conj(z_k)) + (1.0 - self.alpha) * self.power_vector
            )

            y_k = complex(np.vdot(self.w_dft, z_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            denom = self.gamma + self.power_vector
            self.w_dft = self.w_dft + self.step_size * np.conj(e_k) * (z_k / denom)

            self.w = ifft(self.w_dft) * self._sqrtN

            self._record_history()
            w_hist_dft.append(self.w_dft.copy())

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[TDomainDFT] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "coefficients_dft": np.asarray(w_hist_dft),
                "power_vector_last": self.power_vector.copy(),
                "sqrtN": self._sqrtN,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF