# lattice.lrls_priori.py
#
#      Implements the Lattice RLS algorithm based on a priori errors.
#      (Algorithm 7.4 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import perf_counter
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class LRLSPriori(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) using a priori errors.

    Implements Algorithm 7.4 (Diniz) in a lattice (prediction + ladder) structure.

    Library conventions
    -------------------
    - Complex arithmetic (`supports_complex=True`).
    - Ladder coefficients are stored in `self.v` (length M+1).
    - For consistency with the library base class:
        * `self.w` mirrors `self.v`
        * `self._record_history()` is called each iteration
        * coefficients history is available as `result.coefficients`
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Number of lattice sections M. Ladder has M+1 coefficients.
        lambda_factor:
            Forgetting factor λ.
        epsilon:
            Energy initialization / regularization.
        w_init:
            Optional initial ladder coefficient vector (length M+1). If None, zeros.
        denom_floor:
            Floor used to avoid division by (near) zero in normalization terms.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)
        self._tiny = float(denom_floor)

        self.delta = np.zeros(self.n_sections, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

        # Mirror to base API
        self.w = self.v.copy()
        self.w_history = []
        self._record_history()

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes LRLS adaptation (a priori version) over (x[k], d[k]).

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                A priori error e[k].
            coefficients:
                History of ladder coefficients v (mirrored in `self.w_history`).
            error_type:
                "a_priori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["xi_f"], extra["xi_b"], extra["delta"], extra["delta_v"]:
            Final arrays at the end of adaptation.
        """
        t0 = perf_counter()

        # validate_input already normalizes to 1D and matches lengths.
        # Force complex to respect supports_complex=True (even if x/d are real).
        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            alpha_f = complex(x_in[k])

            alpha_b = np.zeros(self.n_sections + 1, dtype=complex)
            alpha_b[0] = x_in[k]

            gamma = 1.0
            gamma_orders = np.ones(self.n_sections + 1, dtype=float)

            # -------------------------
            # Lattice stage (a priori)
            # -------------------------
            for m in range(self.n_sections):
                gamma_orders[m] = gamma
                denom_g = max(gamma, self._tiny)

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(alpha_f)) / denom_g
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + self._tiny)
                kappa_b = self.delta[m] / (self.xi_f[m] + self._tiny)

                alpha_f_next = alpha_f - kappa_f * self.error_b_prev[m]
                alpha_b[m + 1] = self.error_b_prev[m] - kappa_b * alpha_f

                # Energy updates (kept as in your code, with safe denominators)
                self.xi_f[m] = (
                    self.lam * self.xi_f[m]
                    + float(np.real(alpha_f * np.conj(alpha_f))) / denom_g
                )
                self.xi_b[m] = (
                    self.lam * self.xi_b[m]
                    + float(np.real(alpha_b[m] * np.conj(alpha_b[m]))) / denom_g
                )

                denom_xib = self.xi_b[m] + self._tiny
                gamma_next = gamma - (
                    float(np.real(alpha_b[m] * np.conj(alpha_b[m]))) / denom_xib
                )
                gamma = max(gamma_next, self._tiny)
                alpha_f = alpha_f_next

            gamma_orders[self.n_sections] = gamma
            self.xi_f[self.n_sections] = (
                self.lam * self.xi_f[self.n_sections]
                + float(np.real(alpha_f * np.conj(alpha_f))) / max(gamma, self._tiny)
            )
            self.xi_b[self.n_sections] = (
                self.lam * self.xi_b[self.n_sections]
                + float(np.real(alpha_b[self.n_sections] * np.conj(alpha_b[self.n_sections])))
                / max(gamma, self._tiny)
            )

            # -------------------------
            # Ladder stage (a priori)
            # -------------------------
            alpha_e = complex(d_in[k])

            for m in range(self.n_sections + 1):
                denom_go = max(gamma_orders[m], self._tiny)

                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (alpha_b[m] * np.conj(alpha_e)) / denom_go
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)
                alpha_e = alpha_e - np.conj(self.v[m]) * alpha_b[m]

            e_k = alpha_e * gamma
            errors[k] = e_k
            outputs[k] = d_in[k] - e_k

            self.error_b_prev = alpha_b.copy()

            # Mirror ladder coeffs into base API + record history
            self.w = self.v.copy()
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[LRLSPriori] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "xi_f": self.xi_f.copy(),
                "xi_b": self.xi_b.copy(),
                "delta": self.delta.copy(),
                "delta_v": self.delta_v.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF