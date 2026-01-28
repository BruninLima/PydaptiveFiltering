#  set_membership.bnlms.py
#
#       Implements the Set-membership Binormalized LMS algorithm for COMPLEX valued data.
#       (Algorithm 6.5 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@wam@gmail.com
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SMBNLMS(AdaptiveFilter):
    """
    Implements the Set-membership Binormalized LMS (SM-BNLMS) algorithm for complex-valued data.

    This algorithm is a specific case of SM-AP with L=1, designed to improve
    convergence speed over SM-NLMS with low computational overhead by reusing
    the previous regressor. (Algorithm 6.5, Diniz)
    """
    supports_complex: bool = True

    gamma_bar: float
    gamma: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        gamma_bar: float,
        gamma: float,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        gamma_bar:
            Upper bound for the error magnitude (set-membership threshold).
        gamma:
            Regularization factor to avoid division by zero (and stabilize denominator).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.n_coeffs = int(self.filter_order + 1)

        self.regressor_prev: np.ndarray = np.zeros(self.n_coeffs, dtype=complex)

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
        Executes the SM-BNLMS adaptation.

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
                A-priori output y[k] = w^H x_k.
            errors:
                A-priori error e[k] = d[k] - y[k].
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
            Trajectory of the SM step-size factor mu[k] (0 when no update).
        extra["den"]:
            Denominator trajectory used in lambda1/lambda2 (0 when no update).
        extra["lambda1"]:
            Lambda1 trajectory (0 when no update).
        extra["lambda2"]:
            Lambda2 trajectory (0 when no update).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)
        n_coeffs: int = int(self.n_coeffs)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        mu_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        den_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        lam1_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None
        lam2_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        self.n_updates = 0

        self.regressor = np.asarray(self.regressor, dtype=complex)
        if self.regressor.size != n_coeffs:
            self.regressor = np.zeros(n_coeffs, dtype=complex)

        self.regressor_prev = np.asarray(self.regressor_prev, dtype=complex)
        if self.regressor_prev.size != n_coeffs:
            self.regressor_prev = np.zeros(n_coeffs, dtype=complex)

        for k in range(n_samples):
            self.regressor_prev = self.regressor.copy()

            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x[k]

            yk: complex = complex(np.dot(self.w.conj(), self.regressor))
            ek: complex = complex(d[k] - yk)

            outputs[k] = yk
            errors[k] = ek

            eabs: float = float(np.abs(ek))

            if eabs > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True

                mu: float = float(1.0 - (self.gamma_bar / eabs))

                norm_sq: float = float(np.real(np.dot(self.regressor.conj(), self.regressor)))
                prev_norm_sq: float = float(np.real(np.dot(self.regressor_prev.conj(), self.regressor_prev)))
                cross_term: complex = complex(np.dot(self.regressor_prev.conj(), self.regressor))

                den: float = float(self.gamma + (norm_sq * prev_norm_sq) - (np.abs(cross_term) ** 2))

                if den <= 0.0:
                    den = float(self.gamma + 1e-30)

                lambda1: complex = complex((mu * ek * prev_norm_sq) / den)
                lambda2: complex = complex(-(mu * ek * np.conj(cross_term)) / den)

                self.w = self.w + (np.conj(lambda1) * self.regressor) + (np.conj(lambda2) * self.regressor_prev)

                if return_internal_states:
                    if mu_track is not None:
                        mu_track[k] = mu
                    if den_track is not None:
                        den_track[k] = den
                    if lam1_track is not None:
                        lam1_track[k] = lambda1
                    if lam2_track is not None:
                        lam2_track[k] = lambda2
            else:
                if return_internal_states and mu_track is not None:
                    mu_track[k] = 0.0

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-BNLMS] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.03f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "mu": mu_track,
                    "den": den_track,
                    "lambda1": lam1_track,
                    "lambda2": lam2_track,
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
