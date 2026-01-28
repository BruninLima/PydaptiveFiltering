#  set_membership.simplified_puap.py
#
#       Implements the Simplified Set-membership Partial-Update
#       Affine-Projection (SM-Simp-PUAP) algorithm for COMPLEX valued data.
#       (Algorithm 6.6 - book: Adaptive Filtering: Algorithms and Practical
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

import warnings
import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SimplifiedSMPUAP(AdaptiveFilter):
    """
    Implements the Simplified Set-membership Partial-Update Affine-Projection (SM-Simp-PUAP)
    algorithm for complex-valued data. (Algorithm 6.6, Diniz)

    Note
    ----
    The original implementation warns that this algorithm is under development and may be unstable
    for complex-valued simulations.
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
        up_selector: Union[np.ndarray, list],
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
            Regularization factor for the AP correlation matrix.
        L:
            Reuse data factor / constraint length (projection order).
        up_selector:
            Partial-update selector matrix with shape (M+1, N), entries in {0,1}.
            Each column selects which coefficients are updated at iteration k.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        warnings.warn(
            "SM-Simp-PUAP is currently under development and may not produce intended results. "
            "Instability or divergence (high MSE) has been observed in complex-valued simulations.",
            UserWarning,
        )

        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)
        self.n_coeffs = int(self.filter_order + 1)

        sel = np.asarray(up_selector)
        if sel.ndim != 2:
            raise ValueError("up_selector must be a 2D array with shape (M+1, N).")
        if sel.shape[0] != self.n_coeffs:
            raise ValueError(
                f"up_selector must have shape (M+1, N) with M+1={self.n_coeffs}, got {sel.shape}."
            )
        self.up_selector: np.ndarray = sel

        # Regressor matrix: columns are current/past regressors (x_k, x_{k-1}, ..., x_{k-L})
        self.regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)

        # Backwards-compat alias
        self.X_matrix = self.regressor_matrix

        # Bookkeeping
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
        Executes the SM-Simp-PUAP adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime and update count.
        return_internal_states:
            If True, includes internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                A-priori output y[k] (first component of AP output vector).
            errors:
                A-priori error e[k] (first component of AP error vector).
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
        extra["mu"]:
            Trajectory of mu[k] (0 when no update).
        extra["selected_count"]:
            Number of selected coefficients each iteration.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)
        Lp1: int = int(self.L + 1)

        # Validate selector length vs iterations
        if self.up_selector.shape[1] < n_samples:
            raise ValueError(
                f"up_selector has {self.up_selector.shape[1]} columns, but signal has {n_samples} samples."
            )

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        mu_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        selcnt_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=int) if return_internal_states else None

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        # Padding (matches original slicing/indexing)
        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])
        prefixed_desired: np.ndarray = np.concatenate([np.zeros(self.L, dtype=complex), d])

        # u1 = [1, 0, 0, ..., 0]^T  (selects first component)
        u1: np.ndarray = np.zeros((Lp1, 1), dtype=complex)
        u1[0, 0] = 1.0

        for k in range(n_samples):
            # Update regressor matrix
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]
            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            # AP a-priori output/error vectors
            output_ap_conj: np.ndarray = (self.regressor_matrix.conj().T) @ w_current  # (L+1,1)
            desired_slice = prefixed_desired[k + self.L : stop : -1]
            error_ap_conj: np.ndarray = desired_slice.conj().reshape(-1, 1) - output_ap_conj  # (L+1,1)

            yk: complex = complex(output_ap_conj[0, 0])
            ek: complex = complex(error_ap_conj[0, 0])

            outputs[k] = yk
            errors[k] = ek

            eabs: float = float(np.abs(ek))
            if eabs > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True
                mu: float = float(1.0 - (self.gamma_bar / eabs))
            else:
                mu = 0.0

            # Partial-update selector for this iteration (column k): shape (M+1,1)
            c_vec: np.ndarray = self.up_selector[:, k].reshape(-1, 1).astype(float)

            if return_internal_states and selcnt_track is not None:
                selcnt_track[k] = int(np.sum(c_vec != 0))

            if mu > 0.0:
                # Apply selection (element-wise) to regressor matrix
                C_reg: np.ndarray = c_vec * self.regressor_matrix  # (M+1, L+1)

                # R = X^H C X  (as in original: regressor_matrix^H @ C_reg)
                R: np.ndarray = (self.regressor_matrix.conj().T @ C_reg) + self.gamma * np.eye(Lp1)

                rhs: np.ndarray = mu * ek * u1  # (L+1,1)

                try:
                    inv_term = np.linalg.solve(R, rhs)
                except np.linalg.LinAlgError:
                    inv_term = np.linalg.pinv(R) @ rhs

                w_current = w_current + (C_reg @ inv_term)

            if return_internal_states and mu_track is not None:
                mu_track[k] = mu

            # Commit coefficients + history
            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-Simp-PUAP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.2f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "mu": mu_track,
                    "selected_count": selcnt_track,
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