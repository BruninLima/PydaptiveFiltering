#  set_membership.nlms.py
#
#       Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
#       (Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SMNLMS(AdaptiveFilter):
    """
    Implements the Set-membership Normalized LMS algorithm for complex-valued data.

    Coefficients are updated only when |e(k)| > gamma_bar. (Algorithm 6.1, Diniz)
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
            Error magnitude threshold for triggering updates.
        gamma:
            Regularization factor to avoid division by zero in normalization.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)
        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.n_coeffs = int(self.filter_order + 1)

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
        Executes the SM-NLMS adaptation.

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
            Denominator trajectory gamma + ||x_k||^2 (0 when no update).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        mu_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        den_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None

        self.n_updates = 0

        self.regressor = np.asarray(self.regressor, dtype=complex)
        if self.regressor.size != self.n_coeffs:
            self.regressor = np.zeros(self.n_coeffs, dtype=complex)

        for k in range(n_samples):
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
                den: float = float(self.gamma + norm_sq)

                if den <= 0.0:
                    den = float(self.gamma + 1e-30)

                self.w = self.w + (mu / den) * (np.conj(ek) * self.regressor)

                if return_internal_states:
                    if mu_track is not None:
                        mu_track[k] = mu
                    if den_track is not None:
                        den_track[k] = den
            else:
                if return_internal_states and mu_track is not None:
                    mu_track[k] = 0.0

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            pct = (100.0 * self.n_updates / n_samples) if n_samples > 0 else 0.0
            print(f"[SM-NLMS] Updates: {self.n_updates}/{n_samples} ({pct:.1f}%) | Runtime: {runtime_s * 1000:.03f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "mu": mu_track,
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