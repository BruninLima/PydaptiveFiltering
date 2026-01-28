#  rls.rls.py
#
#       Implements the RLS algorithm for COMPLEX valued data.
#       (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import time
from typing import Any, Dict, Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class RLS(AdaptiveFilter):
    """
    Recursive Least Squares (RLS) for complex-valued adaptive FIR filtering.

    Implements Algorithm 5.3 (Diniz). RLS minimizes an exponentially-weighted
    least-squares cost by updating an inverse correlation matrix using the
    matrix inversion lemma.

    Recursion (common form)
    -----------------------
        y[k] = w[k]^H x_k
        e[k] = d[k] - y[k]

        g[k] = (S[k-1] x_k) / (lambda + x_k^H S[k-1] x_k)
        w[k] = w[k-1] + conj(e[k]) g[k]
        S[k] = (S[k-1] - g[k] x_k^H S[k-1]) / lambda

    Notes
    -----
    - Complex-valued implementation (supports_complex=True).
    - By default, returns a priori output/error.
    - If `return_internal_states=True`, includes a posteriori sequences and
      selected internal states in `result.extra`.
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
            Initialization for S_d(0) = (1/delta) * I. Must be positive.
        lamb:
            Forgetting factor λ, typically 0 < λ <= 1.
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
        Run RLS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, includes a posteriori sequences and final internal states in `result.extra`.

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
            A posteriori output y_post[k] computed after updating w.
        extra["errors_posteriori"]:
            A posteriori error e_post[k] = d[k] - y_post[k].
        extra["S_d_last"]:
            Final inverse correlation matrix.
        extra["gain_last"]:
            Last gain vector g.
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

            y_k: complex = complex(np.vdot(self.w, self.regressor))
            e_k: complex = d[k] - y_k

            outputs[k] = y_k
            errors[k] = e_k

            Sx: np.ndarray = self.S_d @ self.regressor
            den: complex = self.lamb + complex(np.vdot(self.regressor, Sx))
            if abs(den) < self._safe_eps:
                den = den + (self._safe_eps + 0.0j)

            g: np.ndarray = Sx / den
            last_gain = g

            self.w = self.w + np.conj(e_k) * g

            self.S_d = (self.S_d - np.outer(g, np.conj(Sx))) / self.lamb

            if return_internal_states:
                yk_post = complex(np.vdot(self.w, self.regressor))
                y_post[k] = yk_post
                e_post[k] = d[k] - yk_post

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[RLS] Completed in {runtime_s * 1000:.03f} ms")

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