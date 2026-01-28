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
    Stabilized Fast Transversal RLS (SFT-RLS) algorithm (real-valued).

    The Stabilized Fast Transversal RLS is a numerically robust variant of the
    Fast Transversal RLS. It preserves the approximately :math:`O(M)` per-sample
    complexity of transversal RLS recursions while improving stability in
    finite-precision arithmetic by introducing feedback stabilization in the
    backward prediction recursion (via ``kappa1``, ``kappa2``, ``kappa3``) and by
    guarding divisions/energies through floors and optional clipping.

    This implementation corresponds to Diniz (Alg. 8.2) and is restricted to
    **real-valued** input/desired sequences (enforced by ``ensure_real_signals``).

    Parameters
    ----------
    filter_order : int
        FIR filter order ``M``. The number of coefficients is ``M + 1``.
    forgetting_factor : float, optional
        Exponential forgetting factor ``lambda``. Default is 0.99.
    epsilon : float, optional
        Positive initialization for the minimum prediction-error energies
        (regularization), used as :math:`\\xi_{\\min}(0)` in the recursions.
        Default is 1e-1.
    kappa1, kappa2, kappa3 : float, optional
        Stabilization constants used to form stabilized versions of the backward
        prediction error. Defaults are 1.5, 2.5, and 1.0.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.
    denom_floor : float, optional
        Safety floor used to clamp denominators before inversion to prevent
        overflow/underflow and non-finite values during internal recursions.
        If None, a small value based on machine ``tiny`` is used.
    xi_floor : float, optional
        Safety floor for prediction error energies (e.g., ``xi_min_f``,
        ``xi_min_b``). If None, a small value based on machine ``tiny`` is used.
    gamma_clip : float, optional
        Optional clipping threshold applied to an intermediate conversion factor
        to avoid extreme values (singularities). If None, no clipping is applied.

    Notes
    -----
    Convention
    ~~~~~~~~~~
    At time ``k``, the internal regressor window has length ``M + 2`` (denoted
    ``r`` in the code) and is formed in reverse order (most recent sample first).
    The main adaptive filter uses the first ``M + 1`` entries of this window.

    A priori vs a posteriori
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The a priori output and error are:

    .. math::
        y(k) = w^T(k-1) x_k, \\qquad e(k) = d(k) - y(k),

    and the a posteriori error returned by this implementation is:

    .. math::
        e_{\\text{post}}(k) = \\gamma(k)\\, e(k),

    where :math:`\\gamma(k)` is produced by the stabilized transversal recursions.

    Stabilization with kappa
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The algorithm forms stabilized backward-error combinations (three variants)
    from two backward-error lines in the recursion (named ``e_b_line1`` and
    ``e_b_line2`` in the code). Conceptually:

    .. math::
        e_{b,i}(k) = \\kappa_i\\, e_{b,2}(k) + (1-\\kappa_i)\\, e_{b,1}(k),

    for :math:`\\kappa_i \\in \\{\\kappa_1, \\kappa_2, \\kappa_3\\}`.

    Numerical safeguards
    ~~~~~~~~~~~~~~~~~~~~
    Several denominators are clamped to ``denom_floor`` before inversion and
    minimum energies are floored by ``xi_floor``. The counts of clamp events are
    tracked and returned in ``extra["clamp_stats"]``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
    Implementation*, 5th ed., Algorithm 8.2.
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
        Executes the stabilized FT-RLS adaptation loop (real-valued).

        Parameters
        ----------
        input_signal : array_like of float
            Real-valued input sequence ``x[k]`` with shape ``(N,)``.
        desired_signal : array_like of float
            Real-valued desired/reference sequence ``d[k]`` with shape ``(N,)``.
            Must have the same length as ``input_signal``.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes internal trajectories in ``result.extra``:
            - ``"xi_min_f"``: ndarray of float, shape ``(N,)`` (forward minimum
              prediction-error energy).
            - ``"xi_min_b"``: ndarray of float, shape ``(N,)`` (backward minimum
              prediction-error energy).
            - ``"gamma"``: ndarray of float, shape ``(N,)`` (conversion factor).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                A priori output sequence ``y[k]``.
            - errors : ndarray of float, shape ``(N,)``
                A priori error sequence ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict
                Always includes:
                - ``"errors_posteriori"``: ndarray of float, shape ``(N,)`` with
                  :math:`e_{\\text{post}}(k)`.
                - ``"clamp_stats"``: dict with counters of denominator clamps.
                Additionally includes ``"xi_min_f"``, ``"xi_min_b"``, and
                ``"gamma"`` if ``return_internal_states=True``.
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

            y_k: float = float(np.dot(self.w, r[:-1]))
            outputs[k] = y_k
            e_k: float = float(d[k] - y_k)
            errors[k] = e_k
            e_post_k: float = float(e_k * gamma_n_3)
            errors_post[k] = e_post_k

            self.w += phi_hat_n * e_post_k
            self._record_history()

            if return_internal_states and xi_f_track is not None:
                xi_f_track[k], xi_b_track[k], gamma_track[k] = xi_min_f, xi_min_b, gamma_n_3

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[StabFastRLS] Completed in {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {"errors_posteriori": errors_post, "clamp_stats": clamp_counter}
        if return_internal_states:
            extra.update({"xi_min_f": xi_f_track, "xi_min_b": xi_b_track, "gamma": gamma_track})

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF