#  lms.sign_data.py
#
#       Implements the Sign-Data LMS algorithm for COMPLEX valued data.
#       (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
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

ArrayLike = Union[np.ndarray, list]


class SignData(AdaptiveFilter):
    """
    Complex Sign-Data LMS adaptive filter.

    Low-complexity LMS variant in which the regressor vector is replaced by its
    element-wise sign. This reduces multiplications (since the update uses a
    ternary/sign regressor), at the expense of slower convergence and/or larger
    steady-state misadjustment in many scenarios.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    At iteration ``k``, form the regressor vector (newest sample first):

    .. math::
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{C}^{M+1}.

    The a priori output and error are

    .. math::
        y[k] = w^H[k] x_k, \\qquad e[k] = d[k] - y[k].

    Define the element-wise sign regressor ``\\operatorname{sign}(x_k)``.
    The update implemented here is

    .. math::
        w[k+1] = w[k] + 2\\mu\\, e^*[k] \\, \\operatorname{sign}(x_k).

    Implementation details
        - For complex inputs, ``numpy.sign`` applies element-wise and returns
          ``x/|x|`` when ``x != 0`` and ``0`` when ``x == 0``.
        - The factor ``2`` in the update matches the implementation in this
          module (consistent with common LMS gradient conventions).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.1 (sign-based LMS variants).
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.step_size = float(step_size)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Sign-Data LMS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal state in ``result.extra``:
            ``"last_sign_regressor"`` (``sign(x_k)``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar output sequence, ``y[k] = w^H[k] x_k``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=complex).ravel()
        d = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(x.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        x_padded = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        last_sign_xk: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]

            y_k = complex(np.vdot(self.w, x_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            sign_xk = np.sign(x_k)
            last_sign_xk = sign_xk

            self.w = self.w + (2.0 * self.step_size) * np.conj(e_k) * sign_xk
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[SignData] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"last_sign_regressor": None if last_sign_xk is None else last_sign_xk.copy()}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF