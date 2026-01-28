# blind.godard.py
#
#       Implements the Godard algorithm for COMPLEX valued data.
#       (Algorithm 13.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace.wam@gmail.com
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class Godard(AdaptiveFilter):
    """
    Godard blind adaptive algorithm (complex-valued).

    The Godard criterion generalizes constant-modulus equalization by using
    exponents ``p`` and ``q`` in a family of dispersion-based cost functions.
    It is commonly used for blind channel equalization and includes CMA(2,2)
    as a special case.

    This implementation follows Diniz (Alg. 13.1) and estimates the dispersion
    constant ``R_q`` directly from the *input sequence* via sample moments.

    Parameters
    ----------
    filter_order : int, optional
        FIR filter order ``M``. The number of coefficients is ``M + 1``.
        Default is 5.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 0.01.
    p_exponent : int, optional
        Exponent ``p`` used in the Godard cost / gradient factor. Default is 2.
    q_exponent : int, optional
        Exponent ``q`` used in the modulus term. Default is 2.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    Let the regressor vector be ``x_k = [x[k], x[k-1], ..., x[k-M]]^T`` and the
    output:

    .. math::
        y(k) = w^H(k) x_k.

    Define the dispersion error (scalar):

    .. math::
        e(k) = |y(k)|^q - R_q.

    In this implementation, the dispersion constant is estimated from the input
    using sample moments:

    .. math::
        R_q \\approx \\frac{\\mathbb{E}[|x|^{2q}]}{\\mathbb{E}[|x|^q]}
        \\approx \\frac{\\frac{1}{N}\\sum_k |x(k)|^{2q}}
                     {\\frac{1}{N}\\sum_k |x(k)|^q},

    with a small ``safe_eps`` to prevent division by zero.

    The instantaneous complex gradient factor is computed as:

    .. math::
        \\phi(k) = p\\,q\\, e(k)^{p-1}\\, |y(k)|^{q-2}\\, y^*(k),

    and the coefficient update used here is:

    .. math::
        w(k+1) = w(k) - \\frac{\\mu}{2}\\, \\phi(k)\\, x_k.

    Numerical stability
    ~~~~~~~~~~~~~~~~~~~
    When ``|y(k)|`` is very small, the term ``|y(k)|^{q-2}`` can be ill-defined
    for ``q < 2`` or can amplify noise. This implementation sets ``phi(k)=0``
    when ``|y(k)| <= safe_eps``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 13.1.
    """

    supports_complex: bool = True
    step_size: float
    p: int
    q: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.01,
        p_exponent: int = 2,
        q_exponent: int = 2,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.p = int(p_exponent)
        self.q = int(q_exponent)
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
        Executes the Godard adaptation loop over an input sequence.

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
            ``"dispersion_constant"`` (estimated ``R_q``) and ``"phi_gradient"``
            (trajectory of ``phi(k)`` with shape ``(N,)``).
        safe_eps : float, optional
            Small epsilon used to avoid division by zero when estimating
            ``R_q`` and to gate the computation of ``phi(k)`` when ``|y(k)|`` is
            close to zero. Default is 1e-12.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Output sequence ``y[k]``.
            - errors : ndarray of float, shape ``(N,)``
                Dispersion error sequence ``e[k] = |y(k)|^q - R_q``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"blind_godard"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        num: float = float(np.mean(np.abs(x) ** (2 * self.q)))
        den: float = float(np.mean(np.abs(x) ** self.q))
        desired_level: float = float(num / (den + safe_eps)) if den > safe_eps else 0.0

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=float)

        phi_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            e_k: float = float((np.abs(y_k) ** self.q) - desired_level)
            errors[k] = e_k

            if np.abs(y_k) > safe_eps:
                phi_k: complex = complex(
                    self.p
                    * self.q
                    * (e_k ** (self.p - 1))
                    * (np.abs(y_k) ** (self.q - 2))
                    * np.conj(y_k)
                )
            else:
                phi_k = 0.0 + 0.0j

            if return_internal_states and phi_track is not None:
                phi_track[k] = phi_k

            self.w = self.w - (self.step_size * phi_k * x_k) / 2.0
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[Godard] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "phi_gradient": phi_track,
                "dispersion_constant": desired_level,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_godard",
            extra=extra,
        )
# EOF