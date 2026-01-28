#  set_membership.affine_projection.py
#
#       Implements the Set-membership Affine-Projection (SM-AP) algorithm
#       for COMPLEX valued data.
#       (Algorithm 6.2 - book: Adaptive Filtering: Algorithms and Practical
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

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SMAffineProjection(AdaptiveFilter):
    """
    Implements the Set-membership Affine-Projection (SM-AP) algorithm for complex-valued data.

    This is a supervised algorithm, i.e., it requires both input_signal and desired_signal.
    """
    supports_complex: bool = True

    gamma_bar: float
    gamma_bar_vector: np.ndarray
    gamma: float
    L: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        gamma_bar: float,
        gamma_bar_vector: Union[np.ndarray, list],
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
            Upper bound for the (a-priori) error magnitude used by set-membership criterion.
        gamma_bar_vector:
            Target a-posteriori error vector, size (L+1,). (Algorithm-dependent)
        gamma:
            Regularization factor for the AP correlation matrix.
        L:
            Reuse data factor / constraint length (projection order).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)

        self.n_coeffs = int(self.filter_order + 1)

        gvec = np.asarray(gamma_bar_vector, dtype=complex).ravel()
        if gvec.size != (self.L + 1):
            raise ValueError(
                f"gamma_bar_vector must have size L+1 = {self.L + 1}, got {gvec.size}"
            )
        self.gamma_bar_vector = gvec.reshape(-1, 1)

        self.regressor_matrix = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)

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
        Executes the SM-AP adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime and update count.
        return_internal_states:
            If True, includes additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                A-priori output y[k].
            errors:
                A-priori error e[k] = d[k] - y[k] (first component of AP error vector).
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
        extra["errors_vector"]:
            Full AP a-priori error vector over time, shape (N, L+1).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)
        Lp1: int = int(self.L + 1)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)
        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        errors_vec_track: Optional[np.ndarray] = (
            np.zeros((n_samples, Lp1), dtype=complex) if return_internal_states else None
        )

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])
        prefixed_desired: np.ndarray = np.concatenate([np.zeros(self.L, dtype=complex), d])

        for k in range(n_samples):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]

            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            output_ap_conj = (self.regressor_matrix.conj().T) @ w_current

            desired_slice = prefixed_desired[k + self.L : stop : -1]
            error_ap_conj = desired_slice.conj().reshape(-1, 1) - output_ap_conj

            yk = output_ap_conj[0, 0]
            ek = error_ap_conj[0, 0]

            outputs[k] = yk
            errors[k] = ek
            if return_internal_states and errors_vec_track is not None:
                errors_vec_track[k, :] = error_ap_conj.ravel()

            if np.abs(ek) > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True

                R = (self.regressor_matrix.conj().T @ self.regressor_matrix) + self.gamma * np.eye(Lp1)
                b = error_ap_conj - self.gamma_bar_vector.conj()

                try:
                    step = np.linalg.solve(R, b)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(R) @ b

                w_current = w_current + (self.regressor_matrix @ step)

            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-AP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra["errors_vector"] = errors_vec_track

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF
