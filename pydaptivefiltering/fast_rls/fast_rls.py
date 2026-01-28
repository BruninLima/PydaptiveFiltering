# fast_rls.fast_rls.py
#
#       Implements the Fast Transversal RLS algorithm for COMPLEX valued data.
#       (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class FastRLS(AdaptiveFilter):
    """
    Implements the Fast Transversal RLS algorithm for complex-valued data.

    This is a supervised algorithm, i.e., it requires both input_signal and desired_signal.
    """
    supports_complex: bool = True

    forgetting_factor: float
    epsilon: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        forgetting_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        forgetting_factor:
            Forgetting factor (lambda), typically close to 1.
        epsilon:
            Regularization / initial prediction error energy (positive).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)
        self.forgetting_factor = float(forgetting_factor)
        self.epsilon = float(epsilon)
        self.n_coeffs = int(filter_order + 1)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Fast Transversal RLS algorithm.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                A-priori output y[k].
            errors:
                A-priori error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_priori".

        Extra (always)
        -------------
        extra["outputs_posteriori"]:
            A-posteriori output sequence.
        extra["errors_posteriori"]:
            A-posteriori error sequence.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["gamma"]:
            Conversion factor trajectory.
        extra["xi_min_f"]:
            Forward prediction minimum error energy trajectory.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal)
        d: np.ndarray = np.asarray(desired_signal)

        n_samples: int = int(x.size)
        m_plus_1: int = int(self.filter_order + 1)

        outputs: np.ndarray = np.zeros(n_samples, dtype=x.dtype)
        errors: np.ndarray = np.zeros(n_samples, dtype=x.dtype)
        outputs_post: np.ndarray = np.zeros(n_samples, dtype=x.dtype)
        errors_post: np.ndarray = np.zeros(n_samples, dtype=x.dtype)

        gamma_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        xi_f_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None

        w_f: np.ndarray = np.zeros(m_plus_1, dtype=x.dtype)
        w_b: np.ndarray = np.zeros(m_plus_1, dtype=x.dtype)
        phi_hat_n: np.ndarray = np.zeros(m_plus_1, dtype=x.dtype)

        gamma_n: float = 1.0
        xi_min_f_prev: float = float(self.epsilon)
        xi_min_b: float = float(self.epsilon)

        x_padded: np.ndarray = np.zeros(n_samples + m_plus_1, dtype=x.dtype)
        x_padded[m_plus_1:] = x

        for k in range(n_samples):
            regressor: np.ndarray = x_padded[k : k + m_plus_1 + 1][::-1]

            e_f_priori: complex = complex(regressor[0] - np.dot(w_f.conj(), regressor[1:]))
            e_f_post: complex = complex(e_f_priori * gamma_n)

            xi_min_f_curr: float = float(
                self.forgetting_factor * xi_min_f_prev + np.real(e_f_priori * np.conj(e_f_post))
            )

            phi_gain: complex = complex(e_f_priori / (self.forgetting_factor * xi_min_f_prev))

            phi_hat_n_plus_1: np.ndarray = np.zeros(m_plus_1 + 1, dtype=x.dtype)
            phi_hat_n_plus_1[1:] = phi_hat_n
            phi_hat_n_plus_1[0] += phi_gain
            phi_hat_n_plus_1[1:] -= phi_gain * w_f

            w_f = w_f + phi_hat_n * np.conj(e_f_post)

            gamma_n_plus_1: float = float((self.forgetting_factor * xi_min_f_prev * gamma_n) / xi_min_f_curr)

            e_b_priori: complex = complex(self.forgetting_factor * xi_min_b * phi_hat_n_plus_1[-1])

            gamma_n = float(
                1.0 / (np.real(1.0 / gamma_n_plus_1 - (phi_hat_n_plus_1[-1] * np.conj(e_b_priori))) + 1e-30)
            )

            e_b_post: complex = complex(e_b_priori * gamma_n)
            xi_min_b = float(self.forgetting_factor * xi_min_b + np.real(e_b_post * np.conj(e_b_priori)))

            phi_hat_n = phi_hat_n_plus_1[:-1] + phi_hat_n_plus_1[-1] * w_b
            w_b = w_b + phi_hat_n * np.conj(e_b_post)

            y_k: complex = complex(np.dot(self.w.conj(), regressor[:-1]))
            outputs[k] = y_k
            errors[k] = d[k] - outputs[k]

            errors_post[k] = errors[k] * gamma_n
            outputs_post[k] = d[k] - errors_post[k]

            self.w = self.w + phi_hat_n * np.conj(errors_post[k])

            if return_internal_states and gamma_track is not None and xi_f_track is not None:
                gamma_track[k] = gamma_n
                xi_f_track[k] = xi_min_f_curr

            xi_min_f_prev = xi_min_f_curr
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[FastRLS] Completed in {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "outputs_posteriori": outputs_post,
            "errors_posteriori": errors_post,
        }
        if return_internal_states:
            extra.update(
                {
                    "gamma": gamma_track,
                    "xi_min_f": xi_f_track,
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