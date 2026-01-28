# blind.sato.py
#
#       Implements the Sato algorithm for COMPLEX valued data.
#       (Algorithm 13.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class Sato(AdaptiveFilter):
    """
    Sato blind adaptive algorithm (complex-valued).

    The Sato criterion is an early blind equalization method particularly
    associated with multilevel PAM/QAM-type signals. It adapts an FIR equalizer
    by pulling the output toward a fixed magnitude level through the complex
    sign function, using a dispersion constant ``zeta``.

    This implementation follows Diniz (Alg. 13.3) and estimates ``zeta`` from
    the *input sequence* via sample moments.

    Parameters
    ----------
    filter_order : int, optional
        FIR filter order ``M``. The number of coefficients is ``M + 1``.
        Default is 5.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 0.01.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    Let the regressor vector be ``x_k = [x[k], x[k-1], ..., x[k-M]]^T`` and the
    output:

    .. math::
        y(k) = w^H(k) x_k.

    Define the complex sign function (unit-circle projection):

    .. math::
        \\mathrm{csgn}(y) =
        \\begin{cases}
        \\dfrac{y}{|y|}, & |y| > 0 \\\\
        0, & |y| = 0
        \\end{cases}

    The Sato error is:

    .. math::
        e(k) = y(k) - \\zeta\\, \\mathrm{csgn}(y(k)).

    The coefficient update used here is:

    .. math::
        w(k+1) = w(k) - \\mu\\, e^*(k)\\, x_k.

    Dispersion constant
    ~~~~~~~~~~~~~~~~~~~
    In this implementation, the dispersion constant is estimated from the input
    using sample moments:

    .. math::
        \\zeta \\approx \\frac{\\mathbb{E}[|x|^2]}{\\mathbb{E}[|x|]}
        \\approx \\frac{\\frac{1}{N}\\sum_k |x(k)|^2}
                     {\\frac{1}{N}\\sum_k |x(k)|},

    with a small ``safe_eps`` to avoid division by zero.

    Numerical stability
    ~~~~~~~~~~~~~~~~~~~
    To avoid instability when ``|y(k)|`` is very small, this implementation
    sets ``csgn(y(k)) = 0`` when ``|y(k)| <= safe_eps``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 13.3.
    """

    supports_complex: bool = True
    step_size: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.01,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Optional[Union[np.ndarray, list]] = None,
        verbose: bool = False,
        return_internal_states: bool = False,
        safe_eps: float = 1e-12,
    ) -> OptimizationResult:
        """
        Executes the Sato adaptation loop over an input sequence.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : None, optional
            Ignored. This is a blind algorithm: no desired reference is used.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes internal signals in ``result.extra``:
            ``"dispersion_constant"`` (estimated ``zeta``) and
            ``"sato_sign_track"`` (trajectory of ``csgn(y(k))`` with shape
            ``(N,)``).
        safe_eps : float, optional
            Small epsilon used to avoid division by zero when estimating
            ``zeta`` and to gate the computation of ``csgn(y(k))`` when ``|y(k)|``
            is close to zero. Default is 1e-12.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Output sequence ``y[k]``.
            - errors : ndarray of complex, shape ``(N,)``
                Sato error sequence ``e[k] = y(k) - zeta*csgn(y(k))``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"blind_sato"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        num: float = float(np.mean(np.abs(x) ** 2))
        den: float = float(np.mean(np.abs(x)))
        dispersion_constant: float = float(num / (den + safe_eps)) if den > safe_eps else 0.0

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        sign_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            mag: float = float(np.abs(y_k))
            sato_sign: complex = (y_k / mag) if mag > safe_eps else (0.0 + 0.0j)

            if return_internal_states and sign_track is not None:
                sign_track[k] = sato_sign

            e_k: complex = y_k - sato_sign * dispersion_constant
            errors[k] = e_k

            self.w = self.w - self.step_size * np.conj(e_k) * x_k
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[Sato] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "sato_sign_track": sign_track,
                "dispersion_constant": dispersion_constant,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_sato",
            extra=extra,
        )
# EOF