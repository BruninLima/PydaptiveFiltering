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
    Alternative RLS (RLS-Alt) for complex-valued adaptive FIR filtering.

    Implements Algorithm 5.4 (Diniz). This variant reduces computational burden by
    using the auxiliary vector:

        psi[k] = S_d[k-1] x_k

    where S_d is the inverse correlation (or inverse deterministic autocorrelation)
    matrix, and x_k is the tapped-delay-line regressor.

    Notes
    -----
    - Complex-valued implementation (supports_complex=True).
    - Returns the **a priori** output and error by default:
        y[k] = w[k]^H x_k
        e[k] = d[k] - y[k]
      and can optionally provide a posteriori sequences in `extra`.
    - Uses unified base API via `@validate_input`.
    """

    supports_complex: bool = True

    lamb: float
    delta: float
    S_d: np.ndarray

    def __init__(
        self,
        filter_order: int,
        delta: float,
        lamb: float,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        delta:
            Initialization factor for S_d(0) = (1/delta) * I. Must be positive.
        lamb:
            Forgetting factor λ. Typically 0 < λ <= 1.
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        safe_eps:
            Small epsilon used to guard denominators.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.lamb = float(lamb)
        if not (0.0 < self.lamb <= 1.0):
            raise ValueError(f"lamb must satisfy 0 < lamb <= 1. Got lamb={self.lamb}.")

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
        Run RLS-Alt adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, includes a posteriori sequences and last internal matrices in `result.extra`.

        Returns
        -------
        OptimizationResult
            outputs:
                A priori output y[k] = w^H x_k.
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                Coefficient history stored in the base class.
            error_type:
                "a_priori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["outputs_posteriori"]:
            A posteriori output y_post[k] using updated w[k+1].
        extra["errors_posteriori"]:
            A posteriori error e_post[k] = d[k] - y_post[k].
        extra["S_d_last"]:
            Final inverse correlation matrix.
        extra["gain_last"]:
            Kalman gain-like vector g at last iteration.
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

            den: complex = self.lamb + complex(np.vdot(self.regressor, psi))
            if abs(den) < self._safe_eps:
                den = den + (self._safe_eps + 0.0j)

            g: np.ndarray = psi / den
            last_gain = g

            self.w = self.w + np.conj(e_k) * g

            self.S_d = (self.S_d - np.outer(g, np.conj(psi))) / self.lamb

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