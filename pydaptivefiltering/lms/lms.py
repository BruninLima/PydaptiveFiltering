#  lms.lms.py
#
#       Implements the Complex LMS algorithm for COMPLEX valued data.
#       (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.typing import ArrayLike


class LMS(AdaptiveFilter):
    """
    Complex Least-Mean Squares (LMS) adaptive filter.

    Standard complex LMS algorithm for adaptive FIR filtering, following Diniz
    (Alg. 3.2). The method performs a stochastic-gradient update using the
    instantaneous a priori error.

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
        y[k] = w^H[k] x_k, \\qquad e[k] = d[k] - y[k],

    and the LMS update is

    .. math::
        w[k+1] = w[k] + \\mu\\, e^*[k] \\, x_k.

    This implementation:
        - uses complex arithmetic (``supports_complex=True``),
        - returns the a priori error ``e[k]``,
        - records coefficient history via the base class.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 3.2.
    """

    supports_complex: bool = True

    step_size: float

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
    ) -> OptimizationResult:
        """
        Executes the LMS adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.

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
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)
        m: int = int(self.filter_order)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + m + 1][::-1]

            y_k: complex = complex(np.vdot(self.w, x_k)) 
            outputs[k] = y_k
    
            e_k: complex = d[k] - y_k
            errors[k] = e_k

            self.w = self.w + self.step_size * np.conj(e_k) * x_k

            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[LMS] Completed in {runtime_s * 1000:.03f} ms")

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
        )
# EOF