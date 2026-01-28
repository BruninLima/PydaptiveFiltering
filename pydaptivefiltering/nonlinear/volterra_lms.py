#  nonlinear.volterra_lms.py
#
#       Implements the Volterra LMS algorithm for REAL valued data.
#       (Algorithm 11.1 - book: Adaptive Filtering: Algorithms and Practical
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

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class VolterraLMS(AdaptiveFilter):
    """
    Volterra LMS (2nd-order) for real-valued adaptive filtering.

    Implements Algorithm 11.1 (Diniz) using a second-order Volterra expansion.

    For a linear memory length L (called `memory`), the regressor is composed of:
    - linear terms:  [x[k], x[k-1], ..., x[k-L+1]]
    - quadratic terms (with i <= j):
        [x[k]^2, x[k]x[k-1], ..., x[k-L+1]^2]

    Total number of coefficients:
        n_coeffs = L + L(L+1)/2

    Notes
    -----
    - Real-valued only: enforced by `ensure_real_signals`.
    - The base class coefficient vector `self.w` corresponds to the Volterra
      coefficient vector (linear + quadratic). The history returned in
      `OptimizationResult.coefficients` is the stacked trajectory of `self.w`.
    - `step` can be:
        * scalar (same step for all coefficients), or
        * vector (shape (n_coeffs,)) allowing per-term step scaling.
    """

    supports_complex: bool = False

    def __init__(
        self,
        memory: int = 3,
        step: Union[float, np.ndarray, list] = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        memory:
            Linear memory length L. Determines the Volterra regressor size:
            n_coeffs = L + L(L+1)/2.
        step:
            Step-size mu. Can be a scalar or a vector of length n_coeffs.
        w_init:
            Optional initial coefficients (length n_coeffs). If None, zeros.
        safe_eps:
            Small epsilon used for internal safety checks (kept for consistency).
        """
        memory = int(memory)
        if memory <= 0:
            raise ValueError(f"memory must be > 0. Got {memory}.")

        self.memory: int = memory
        self.n_coeffs: int = memory + (memory * (memory + 1)) // 2
        self._safe_eps: float = float(safe_eps)

        super().__init__(filter_order=self.n_coeffs - 1, w_init=w_init)

        if isinstance(step, (list, np.ndarray)):
            step_vec = np.asarray(step, dtype=np.float64).reshape(-1)
            if step_vec.size != self.n_coeffs:
                raise ValueError(
                    f"step vector must have length {self.n_coeffs}, got {step_vec.size}."
                )
            self.step: Union[float, np.ndarray] = step_vec
        else:
            self.step = float(step)

        self.w = np.asarray(self.w, dtype=np.float64)

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
            raise ValueError(
                f"x_lin must have length {self.memory}, got {x_lin.size}."
            )

        quad = np.empty((self.memory * (self.memory + 1)) // 2, dtype=np.float64)
        idx = 0
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad[idx] = x_lin[i] * x_lin[j]
                idx += 1

        return np.concatenate([x_lin, quad], axis=0)

    @ensure_real_signals
    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Volterra LMS adaptation over (x[k], d[k]).

        Parameters
        ----------
        input_signal:
            Input sequence x[k], shape (N,).
        desired_signal:
            Desired sequence d[k], shape (N,).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns selected internal values in `result.extra`.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k] (a priori).
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of Volterra coefficient vector w (stacked from base history).
            error_type:
                "a_priori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["last_regressor"]:
            Last Volterra regressor u[k] (length n_coeffs).
        extra["memory"]:
            Linear memory length L.
        extra["n_coeffs"]:
            Number of Volterra coefficients.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        if x.size != d.size:
            raise ValueError(f"Inconsistent lengths: input({x.size}) != desired({d.size})")
        n_samples = int(x.size)

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        L = int(self.memory)
        x_padded = np.zeros(n_samples + (L - 1), dtype=np.float64)
        x_padded[L - 1 :] = x

        last_u: Optional[np.ndarray] = None

        for k in range(n_samples):
            x_lin = x_padded[k : k + L][::-1]
            u = self._create_volterra_regressor(x_lin)
            last_u = u

            y_k = float(np.dot(self.w, u))
            outputs[k] = y_k

            e_k = float(d[k] - y_k)
            errors[k] = e_k

            if isinstance(self.step, np.ndarray):
                self.w = self.w + (2.0 * self.step) * e_k * u
            else:
                self.w = self.w + (2.0 * float(self.step)) * e_k * u

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[VolterraLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "last_regressor": None if last_u is None else last_u.copy(),
                "memory": int(self.memory),
                "n_coeffs": int(self.n_coeffs),
            }

        return self._pack_results(
            outputs=outputs,
            errors= errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF