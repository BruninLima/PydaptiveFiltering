#  lms.power2_error.py
#
#       Implements the Power-of-Two Error LMS algorithm for REAL valued data.
#       (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class Power2ErrorLMS(AdaptiveFilter):
    """
    Power-of-Two Error LMS adaptive filter (real-valued).

    LMS variant in which the instantaneous a priori error is quantized to a
    power-of-two level (with special cases for large and very small errors),
    aiming to reduce computational complexity in fixed-point / low-cost
    implementations.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    bd : int
        Word length (number of bits) used to define the small-error threshold
        ``2^{-bd+1}``.
    tau : float
        Gain factor applied when ``|e[k]|`` is very small (below ``2^{-bd+1}``).
    step_size : float, optional
        Adaptation step size ``mu``. Default is 1e-2.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Signal model and LMS update
        Let the regressor vector be

        .. math::
            x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T \\in \\mathbb{R}^{M+1},

        with output and a priori error

        .. math::
            y[k] = w^T[k] x_k, \\qquad e[k] = d[k] - y[k].

        The update uses a quantized error ``q(e[k])``:

        .. math::
            w[k+1] = w[k] + 2\\mu\\, q(e[k])\\, x_k.

    Error quantization (as implemented)
        Define the small-error threshold

        .. math::
            \\epsilon = 2^{-bd+1}.

        Then the quantizer is

        .. math::
            q(e) =
            \\begin{cases}
                \\operatorname{sign}(e), & |e| \\ge 1, \\\\
                \\tau\\,\\operatorname{sign}(e), & |e| < \\epsilon, \\\\
                2^{\\lfloor \\log_2(|e|) \\rfloor}\\,\\operatorname{sign}(e),
                & \\text{otherwise.}
            \\end{cases}

        Note that ``numpy.sign(0) = 0``; therefore if ``e[k] == 0`` then
        ``q(e[k]) = 0`` and the update is null.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.1 (modified complexity-reduced LMS variants).
    """

    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int,
        bd: int,
        tau: float,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.bd = int(bd)
        self.tau = float(tau)
        self.step_size = float(step_size)

        if self.bd <= 0:
            raise ValueError(f"bd must be a positive integer. Got bd={self.bd}.")

    @validate_input
    @ensure_real_signals
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Power-of-Two Error LMS adaptation loop over paired sequences.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"last_quantized_error"`` (``q(e[k])``) and ``"small_threshold"``
            (``2^{-bd+1}``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar output sequence, ``y[k] = w^T[k] x_k``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(x.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        x_padded = np.zeros(n_samples + m, dtype=np.float64)
        x_padded[m:] = x

        last_qe: Optional[float] = None
        small_thr = 2.0 ** (-self.bd + 1)

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]

            y_k = float(np.dot(self.w, x_k))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            abs_error = abs(e_k)
            if abs_error >= 1.0:
                qe = float(np.sign(e_k))
            elif abs_error < small_thr:
                qe = float(self.tau * np.sign(e_k))
            else:
                qe = float((2.0 ** np.floor(np.log2(abs_error))) * np.sign(e_k))

            last_qe = qe

            self.w = self.w + (2.0 * self.step_size) * qe * x_k
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[Power2ErrorLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {"last_quantized_error": last_qe, "small_threshold": float(small_thr)}

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF