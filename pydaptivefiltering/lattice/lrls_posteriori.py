# lattice.lrls_posteriori.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors.
#      (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical
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


class LRLSPosteriori(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) using a posteriori errors.

    Implements Algorithm 7.1 (Diniz) in a lattice structure (prediction + ladder).

    Library conventions
    -------------------
    - Complex-valued implementation (`supports_complex=True`).
    - Ladder coefficients are stored in `self.v` (length M+1).
    - `self.w` mirrors `self.v` and history is recorded via `_record_history()`.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: float = 1e-12,
        xi_floor: Optional[float] = None,
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
        xi_floor:
            Floor used to keep energies positive (defaults to epsilon).
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)

        self._tiny = float(denom_floor)
        self._xi_floor = float(xi_floor) if xi_floor is not None else float(self.epsilon)

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
        Executes LRLS adaptation (a posteriori version) over (x[k], d[k]).

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                A posteriori error e[k].
            coefficients:
                History of ladder coefficients v (mirrored in self.w_history).
            error_type:
                "a_posteriori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["xi_f"], extra["xi_b"], extra["delta"], extra["delta_v"]:
            Final arrays at the end of adaptation.
        """
        t0 = perf_counter()

        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors  = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            err_f = complex(x_in[k])

            curr_err_b = np.zeros(self.n_sections + 1, dtype=complex)
            curr_err_b[0] = x_in[k]

            energy_x = float(np.real(err_f * np.conj(err_f)))
            self.xi_f[0] = max(self.lam * self.xi_f[0] + energy_x, self._xi_floor)
            self.xi_b[0] = self.xi_f[0]

            gamma_m = 1.0

            for m in range(self.n_sections):
                denom_g = max(gamma_m, self._tiny)

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(err_f)) / denom_g
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + self._tiny)
                kappa_b = self.delta[m] / (self.xi_f[m] + self._tiny)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_err_b[m + 1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m + 1] = max(
                    self.lam * self.xi_f[m + 1]
                    + float(np.real(new_err_f * np.conj(new_err_f))) / denom_g,
                    self._xi_floor,
                )
                self.xi_b[m + 1] = max(
                    self.lam * self.xi_b[m + 1]
                    + float(np.real(curr_err_b[m + 1] * np.conj(curr_err_b[m + 1]))) / denom_g,
                    self._xi_floor,
                )

                denom_xib = self.xi_b[m] + self._tiny
                energy_b_curr = float(np.real(curr_err_b[m] * np.conj(curr_err_b[m])))
                gamma_m_next = gamma_m - (energy_b_curr / denom_xib)

                gamma_m = max(gamma_m_next, self._tiny)
                err_f = new_err_f

            e_post = complex(d_in[k])
            gamma_ladder = 1.0

            for m in range(self.n_sections + 1):
                denom_gl = max(gamma_ladder, self._tiny)

                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (curr_err_b[m] * np.conj(e_post)) / denom_gl
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)

                e_post = e_post - np.conj(self.v[m]) * curr_err_b[m]

                denom_xib_m = self.xi_b[m] + self._tiny
                energy_b_l = float(np.real(curr_err_b[m] * np.conj(curr_err_b[m])))
                gamma_ladder_next = gamma_ladder - (energy_b_l / denom_xib_m)
                gamma_ladder = max(gamma_ladder_next, self._tiny)

            outputs[k] = d_in[k] - e_post
            errors[k] = e_post

            self.error_b_prev = curr_err_b.copy()

            self.w = self.v.copy()
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[LRLSPosteriori] Completed in {runtime_s * 1000:.02f} ms")

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
            error_type="a_posteriori",
            extra=extra,
        )
# EOF