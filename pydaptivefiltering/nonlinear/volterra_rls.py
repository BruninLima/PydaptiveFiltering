#  nonlinear.volterra_rls.py
#
#       Implements the Volterra RLS algorithm for REAL valued data.
#       (Algorithm 11.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class VolterraRLS(AdaptiveFilter):
    """
    Volterra RLS (2nd-order) for real-valued adaptive filtering.

    Implements Algorithm 11.2 (Diniz) using a second-order Volterra expansion
    and an RLS update on the expanded regressor.

    For linear memory length L (`memory`), the Volterra regressor is:
    - linear terms:  [x[k], x[k-1], ..., x[k-L+1]]
    - quadratic terms (i <= j):
        [x[k]^2, x[k]x[k-1], ..., x[k-L+1]^2]

    Total number of coefficients:
        n_coeffs = L + L(L+1)/2

    Notes
    -----
    - Real-valued only (enforced by `ensure_real_signals`).
    - We return the *a priori* error by default:
        e[k] = d[k] - y[k]  with y[k] = w^T u[k]  (before the weight update)
      and set `error_type="a_priori"`.
    - If `return_internal_states=True`, we also include posterior sequences in `extra`.
    """

    supports_complex: bool = False

    def __init__(
        self,
        memory: int = 3,
        forgetting_factor: float = 0.98,
        delta: float = 1.0,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        memory:
            Linear memory length L. Determines number of Volterra coefficients:
            n_coeffs = L + L(L+1)/2.
        forgetting_factor:
            Forgetting factor λ (typically close to 1). Must satisfy 0 < λ <= 1.
        delta:
            Positive regularization for initializing the inverse correlation matrix:
            P[0] = I / delta.
        w_init:
            Optional initial coefficient vector (length n_coeffs). If None, zeros.
        safe_eps:
            Small epsilon to guard denominators.
        """
        memory = int(memory)
        if memory <= 0:
            raise ValueError(f"memory must be > 0. Got {memory}.")

        lam = float(forgetting_factor)
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"forgetting_factor must satisfy 0 < λ <= 1. Got λ={lam}.")

        delta = float(delta)
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0. Got delta={delta}.")

        self.memory: int = memory
        self.lam: float = lam
        self._safe_eps: float = float(safe_eps)

        self.n_coeffs: int = memory + (memory * (memory + 1)) // 2

        super().__init__(filter_order=self.n_coeffs - 1, w_init=w_init)

        self.w = np.asarray(self.w, dtype=np.float64)

        if w_init is not None:
            w0 = np.asarray(w_init, dtype=np.float64).reshape(-1)
            if w0.size != self.n_coeffs:
                raise ValueError(
                    f"w_init must have length {self.n_coeffs}, got {w0.size}."
                )
            self.w = w0.copy()

        self.P: np.ndarray = (np.eye(self.n_coeffs, dtype=np.float64) / delta)

        self.w_history = []
        self._record_history()

    def _create_volterra_regressor(self, x_lin: np.ndarray) -> np.ndarray:
        """
        Construct the 2nd-order Volterra regressor from a linear delay line.

        Parameters
        ----------
        x_lin:
            Linear delay line of length `memory` ordered as:
            [x[k], x[k-1], ..., x[k-L+1]].

        Returns
        -------
        np.ndarray
            Volterra regressor u[k] of length n_coeffs:
            [linear terms, quadratic terms (i<=j)].
        """
        x_lin = np.asarray(x_lin, dtype=np.float64).reshape(-1)
        if x_lin.size != self.memory:
            raise ValueError(f"x_lin must have length {self.memory}, got {x_lin.size}.")

        quad = np.empty((self.memory * (self.memory + 1)) // 2, dtype=np.float64)
        idx = 0
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad[idx] = x_lin[i] * x_lin[j]
                idx += 1

        return np.concatenate([x_lin, quad], axis=0)

    @ensure_real_signals
    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Volterra RLS adaptation over (x[k], d[k]).

        Parameters
        ----------
        input_signal:
            Input sequence x[k], shape (N,).
        desired_signal:
            Desired sequence d[k], shape (N,).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, includes additional sequences in `result.extra`.

        Returns
        -------
        OptimizationResult
            outputs:
                A priori output y[k] = w^T u[k].
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of Volterra coefficients w (stacked from base history).
            error_type:
                "a_priori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["posteriori_outputs"]:
            Output after the weight update (y_post).
        extra["posteriori_errors"]:
            Error after the weight update (e_post).
        extra["last_gain"]:
            Last RLS gain vector k (shape (n_coeffs,)).
        extra["last_den"]:
            Last denominator (scalar).
        extra["last_regressor"]:
            Last Volterra regressor u[k].
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(x.size)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        y_post = np.zeros(n_samples, dtype=np.float64)
        e_post = np.zeros(n_samples, dtype=np.float64)

        L = int(self.memory)
        x_padded = np.zeros(n_samples + (L - 1), dtype=np.float64)
        x_padded[L - 1 :] = x

        last_k: Optional[np.ndarray] = None
        last_den: Optional[float] = None
        last_u: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_lin = x_padded[k : k + L][::-1]
            u = self._create_volterra_regressor(x_lin)
            last_u = u

            y_k = float(np.dot(self.w, u))
            e_k = float(d[k] - y_k)
            outputs[k] = y_k
            errors[k] = e_k

            Pu = self.P @ u
            den = float(self.lam + np.dot(u, Pu))
            if abs(den) < self._safe_eps:
                den = float(den + np.sign(den) * self._safe_eps) if den != 0.0 else float(self._safe_eps)

            k_gain = Pu / den
            last_k = k_gain
            last_den = den

            self.w = self.w + k_gain * e_k

            self.P = (self.P - np.outer(k_gain, Pu)) / self.lam

            yk_post = float(np.dot(self.w, u))
            ek_post = float(d[k] - yk_post)
            y_post[k] = yk_post
            e_post[k] = ek_post

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[VolterraRLS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "posteriori_outputs": y_post,
                "posteriori_errors": e_post,
                "last_gain": None if last_k is None else last_k.copy(),
                "last_den": last_den,
                "last_regressor": None if last_u is None else last_u.copy(),
                "memory": int(self.memory),
                "n_coeffs": int(self.n_coeffs),
                "forgetting_factor": float(self.lam),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF