# fast_rls.stab_fast_rls.py
#
#       Implements the Stabilized Fast Transversal RLS algorithm for REAL valued data.
#       (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical
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
from pydaptivefiltering.utils.validation import ensure_real_signals


class StabFastRLS(AdaptiveFilter):
    """
    Implements the Stabilized Fast Transversal RLS algorithm for real-valued data.
    """
    supports_complex: bool = False

    lambda_: float
    epsilon: float
    kappa1: float
    kappa2: float
    kappa3: float
    denom_floor: float
    xi_floor: float
    gamma_clip: Optional[float]
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        forgetting_factor: float = 0.99,
        epsilon: float = 1e-1,
        kappa1: float = 1.5,
        kappa2: float = 2.5,
        kappa3: float = 1.0,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: Optional[float] = None,
        xi_floor: Optional[float] = None,
        gamma_clip: Optional[float] = None,
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
        kappa1, kappa2, kappa3:
            Stabilization parameters from the stabilized FTRLS formulation.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        denom_floor:
            Floor for denominators used in safe inversions. If None, a tiny float-based default is used.
        xi_floor:
            Floor for prediction error energies. If None, a tiny float-based default is used.
        gamma_clip:
            Optional clipping threshold for gamma (if provided).
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.filter_order = int(filter_order)
        self.n_coeffs = int(self.filter_order + 1)

        self.lambda_ = float(forgetting_factor)
        self.epsilon = float(epsilon)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.kappa3 = float(kappa3)

        finfo = np.finfo(np.float64)
        self.denom_floor = float(denom_floor) if denom_floor is not None else float(finfo.tiny * 1e3)
        self.xi_floor = float(xi_floor) if xi_floor is not None else float(finfo.tiny * 1e6)
        self.gamma_clip = float(gamma_clip) if gamma_clip is not None else None

        self.w = np.asarray(self.w, dtype=np.float64)

    @staticmethod
    def _clamp_denom(den: float, floor: float) -> float:
        if (not np.isfinite(den)) or (abs(den) < floor):
            return float(np.copysign(floor, den if den != 0 else 1.0))
        return float(den)

    def _safe_inv(self, den: float, floor: float, clamp_counter: Dict[str, int], key: str) -> float:
        den_clamped = self._clamp_denom(den, floor)
        if den_clamped != den:
            clamp_counter[key] = clamp_counter.get(key, 0) + 1
        return 1.0 / den_clamped

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
        Executes the Stabilized Fast Transversal RLS algorithm.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns internal trajectories and clamping stats in result.extra.

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
        extra["errors_posteriori"]:
            A-posteriori error sequence e_post[k] = gamma[k] * e[k].
        extra["clamp_stats"]:
            Dictionary with counters of how many times each denominator was clamped.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["xi_min_f"]:
            Forward prediction error energy trajectory.
        extra["xi_min_b"]:
            Backward prediction error energy trajectory.
        extra["gamma"]:
            Conversion factor trajectory.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=np.float64)
        d: np.ndarray = np.asarray(desired_signal, dtype=np.float64)

        n_samples: int = int(x.size)
        n_taps: int = int(self.filter_order + 1)
        reg_len: int = int(self.filter_order + 2)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.float64)
        errors_post: np.ndarray = np.zeros(n_samples, dtype=np.float64)

        xi_min_f: float = float(self.epsilon)
        xi_min_b: float = float(self.epsilon)
        gamma_n_3: float = 1.0

        xi_f_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=np.float64) if return_internal_states else None
        xi_b_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=np.float64) if return_internal_states else None
        gamma_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=np.float64) if return_internal_states else None

        w_f: np.ndarray = np.zeros(n_taps, dtype=np.float64)
        w_b: np.ndarray = np.zeros(n_taps, dtype=np.float64)
        phi_hat_n: np.ndarray = np.zeros(n_taps, dtype=np.float64)
        phi_hat_np1: np.ndarray = np.zeros(reg_len, dtype=np.float64)

        x_padded: np.ndarray = np.zeros(n_samples + n_taps, dtype=np.float64)
        x_padded[n_taps:] = x

        clamp_counter: Dict[str, int] = {}

        for k in range(n_samples):
            r: np.ndarray = x_padded[k : k + reg_len][::-1]

            e_f_priori: float = float(r[0] - np.dot(w_f, r[1:]))
            e_f_post: float = float(e_f_priori * gamma_n_3)

            scale: float = self._safe_inv(self.lambda_ * xi_min_f, self.denom_floor, clamp_counter, "inv_lam_xi_f")
            phi_hat_np1[0] = scale * e_f_priori
            phi_hat_np1[1:] = phi_hat_n - phi_hat_np1[0] * w_f

            inv_g3: float = self._safe_inv(gamma_n_3, self.denom_floor, clamp_counter, "inv_g3")
            gamma_np1_1: float = self._safe_inv(
                inv_g3 + phi_hat_np1[0] * e_f_priori, self.denom_floor, clamp_counter, "inv_g_np1"
            )

            if self.gamma_clip is not None:
                gamma_np1_1 = float(np.clip(gamma_np1_1, -self.gamma_clip, self.gamma_clip))

            inv_xi_f_lam: float = self._safe_inv(
                xi_min_f * self.lambda_, self.denom_floor, clamp_counter, "inv_xi_f"
            )
            xi_min_f = max(
                self._safe_inv(
                    inv_xi_f_lam - gamma_np1_1 * (phi_hat_np1[0] ** 2),
                    self.denom_floor,
                    clamp_counter,
                    "inv_den_xi_f",
                ),
                self.xi_floor,
            )

            w_f += phi_hat_n * e_f_post

            e_b_line1: float = float(self.lambda_ * xi_min_b * phi_hat_np1[-1])
            e_b_line2: float = float(r[-1] - np.dot(w_b, r[:-1]))

            eb3_1: float = float(e_b_line2 * self.kappa1 + e_b_line1 * (1.0 - self.kappa1))
            eb3_2: float = float(e_b_line2 * self.kappa2 + e_b_line1 * (1.0 - self.kappa2))
            eb3_3: float = float(e_b_line2 * self.kappa3 + e_b_line1 * (1.0 - self.kappa3))

            inv_g_np1_1: float = self._safe_inv(gamma_np1_1, self.denom_floor, clamp_counter, "inv_g_np1_1")
            gamma_n_2: float = self._safe_inv(
                inv_g_np1_1 - phi_hat_np1[-1] * eb3_3, self.denom_floor, clamp_counter, "inv_g_n2"
            )

            xi_min_b = max(
                float(self.lambda_ * xi_min_b + (eb3_2 * gamma_n_2) * eb3_2),
                self.xi_floor,
            )

            phi_hat_n = phi_hat_np1[:-1] + phi_hat_np1[-1] * w_b
            w_b += phi_hat_n * (eb3_1 * gamma_n_2)

            gamma_n_3 = self._safe_inv(
                1.0 + float(np.dot(phi_hat_n, r[:-1])),
                self.denom_floor,
                clamp_counter,
                "inv_g_n3",
            )

            if return_internal_states and xi_f_track is not None and xi_b_track is not None and gamma_track is not None:
                xi_f_track[k] = xi_min_f
                xi_b_track[k] = xi_min_b
                gamma_track[k] = gamma_n_3

            y_k: float = float(np.dot(self.w, r[:-1]))
            outputs[k] = y_k

            e_k: float = float(d[k] - y_k)
            errors[k] = e_k

            e_post_k: float = float(e_k * gamma_n_3)
            errors_post[k] = e_post_k

            self.w += phi_hat_n * e_post_k
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[StabFastRLS] Completed in {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "errors_posteriori": errors_post,
            "clamp_stats": clamp_counter,
        }
        if return_internal_states:
            extra.update(
                {
                    "xi_min_f": xi_f_track,
                    "xi_min_b": xi_b_track,
                    "gamma": gamma_track,
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