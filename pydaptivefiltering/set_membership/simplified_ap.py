#  set_membership.simplified_ap.py
#
#       Implements the Simplified Set-membership Affine-Projection (SM-Simp-AP)
#       algorithm for COMPLEX valued data.
#       (Algorithm 6.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SimplifiedSMAP(AdaptiveFilter):
    """
    Implements the Simplified Set-membership Affine-Projection (SM-Simp-AP) algorithm
    for complex-valued data. (Algorithm 6.3, Diniz)

    In this simplified version, only the current regressor column (x_k) is used
    in the update, while the algorithm still keeps an AP-style regressor matrix
    for compatibility/inspection.
    """
    supports_complex: bool = True
    gamma_bar: float
    gamma: float
    L: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        gamma_bar: float,
        gamma: float,
        L: int,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        gamma_bar:
            Error magnitude threshold for triggering updates.
        gamma:
            Regularization factor in the normalization denominator.
        L:
            Reuse data factor / constraint length (kept for matrix size L+1).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)
        self.n_coeffs = int(self.filter_order + 1)

        self.regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)

        self.X_matrix = self.regressor_matrix

        self.n_updates: int = 0

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the SM-Simp-AP adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime and update stats.
        return_internal_states:
            If True, includes additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                A-priori output y[k] = w^H x_k.
            errors:
                A-priori error e[k] = d[k] - y[k] (same semantics as your original code).
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_priori".

        Extra (always)
        -------------
        extra["n_updates"]:
            Number of coefficient updates (iterations where |e(k)| > gamma_bar).
        extra["update_mask"]:
            Boolean array marking which iterations performed updates.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["step_factor"]:
            Trajectory of the scalar (1 - gamma_bar/|e|) * e used in the update (0 when no update).
        extra["den"]:
            Denominator trajectory (gamma + ||x_k||^2) (0 when no update).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        step_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None
        den_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])

        for k in range(n_samples):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]

            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            xk: np.ndarray = self.regressor_matrix[:, 0:1]

            output_k: complex = complex((xk.conj().T @ w_current).item())
            error_k: complex = complex(np.conj(d[k]) - output_k)

            outputs[k] = output_k
            errors[k] = error_k

            eabs: float = float(np.abs(error_k))

            if eabs > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True

                step_factor: complex = complex((1.0 - (self.gamma_bar / eabs)) * error_k)

                norm_sq: float = float(np.real((xk.conj().T @ xk).item()))
                den: float = float(self.gamma + norm_sq)
                if den <= 0.0:
                    den = float(self.gamma + 1e-30)

                w_current = w_current + (step_factor / den) * xk

                if return_internal_states:
                    if step_track is not None:
                        step_track[k] = step_factor
                    if den_track is not None:
                        den_track[k] = den
            else:
                if return_internal_states:
                    if step_track is not None:
                        step_track[k] = 0.0 + 0.0j
                    if den_track is not None:
                        den_track[k] = 0.0

            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-Simp-AP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.2f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "step_factor": step_track,
                    "den": den_track,
                }
            )

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF