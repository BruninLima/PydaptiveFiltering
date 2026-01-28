# blind.constant_modulus.py
#
#       Implements the Constant-Modulus algorithm for COMPLEX valued data.
#       (Algorithm 13.2 - book: Adaptive Filtering: Algorithms and Practical
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


class CMA(AdaptiveFilter):
    """
    Constant-Modulus Algorithm (CMA) for blind adaptive filtering (complex-valued).

    The CMA adapts an FIR equalizer to produce an output with (approximately)
    constant modulus, making it useful for blind equalization of constant-envelope
    and near-constant-envelope modulations (e.g., PSK and some QAM regimes).

    This implementation follows Diniz (Alg. 13.2) using the classical CMA(2,2)
    instantaneous gradient approximation.

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
    filter output:

    .. math::
        y(k) = w^H(k) x_k.

    CMA(2,2) is commonly derived from minimizing the instantaneous cost:

    .. math::
        J(k) = \\left(|y(k)|^2 - R_2\\right)^2,

    where ``R2`` is the dispersion constant. Using an instantaneous gradient
    approximation, define the scalar error:

    .. math::
        e(k) = |y(k)|^2 - R_2,

    and the (complex) gradient factor:

    .. math::
        \\phi(k) = 2\\, e(k)\\, y^*(k).

    The coefficient update is then:

    .. math::
        w(k+1) = w(k) - \\mu\\, \\phi(k)\\, x_k.

    Dispersion constant
    ~~~~~~~~~~~~~~~~~~~
    In theory, ``R2`` depends on the source constellation statistics and is
    often written as:

    .. math::
        R_2 = \\frac{\\mathbb{E}[|s(k)|^4]}{\\mathbb{E}[|s(k)|^2]}.

    In practice, when the source ``s(k)`` is not available (blind setting),
    ``R2`` is typically chosen from prior knowledge of the modulation or
    estimated from a proxy sequence. If this implementation estimates ``R2``
    from data, it should specify which sequence is used (e.g., input vs output).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 13.2.
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
        Executes the CMA adaptation loop over an input sequence.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : None, optional
            Ignored. This is a blind algorithm: it does not require a desired
            reference signal.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes internal quantities in ``result.extra`` (e.g.,
            the dispersion constant ``R2`` and/or the last/trajectory of
            ``phi(k)`` depending on the implementation).
        safe_eps : float, optional
            Small epsilon used to avoid division by zero if ``R2`` is estimated
            from sample moments. Default is 1e-12.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Output sequence ``y[k]``.
            - errors : ndarray of float or complex, shape ``(N,)``
                CMA error sequence ``e[k] = |y(k)|^2 - R2`` (usually real-valued).
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"blind_constant_modulus"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=float)
        
        denom: float = float(np.mean(np.abs(x) ** 2))
        if denom < safe_eps:
            desired_level: float = 0.0
        else:
            desired_level = float(np.mean(np.abs(x) ** 4) / (denom + safe_eps))

        phi_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            e_k: float = float((np.abs(y_k) ** 2) - desired_level)
            errors[k] = e_k

            phi_k: complex = complex(2.0 * e_k * np.conj(y_k))
            if return_internal_states and phi_track is not None:
                phi_track[k] = phi_k

            self.w = self.w - self.step_size * phi_k * x_k
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[CMA] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "dispersion_constant": desired_level,
                "instantaneous_phi": phi_track,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_constant_modulus",
            extra=extra,
        )
# EOF