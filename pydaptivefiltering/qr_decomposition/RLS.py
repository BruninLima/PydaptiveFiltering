# qr_decomposition.rls.py
#
#       Implements the QR-RLS algorithm for REAL valued data.
#       (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, 3rd Ed., Diniz)
#
#       Notes:
#       - This implementation follows the reference MATLAB code provided:
#         QR_RLS.m (REAL-valued) with the same state variables:
#           ULineMatrix, dLine_q2, regressor, d_line, gamma, and the same
#         “Givens rotation on stacked [reg; U] and [d_line; dLine_q2]”.
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermeodeoliveirapinto@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                               diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class QRRLS(AdaptiveFilter):
    """
    QR-RLS (real-valued) using Givens rotations.

    Implements Algorithm 9.1 (Diniz, 3rd ed.) in a QR-decomposition framework.
    This version mirrors the provided MATLAB routine (QR_RLS.m) and keeps the
    same internal state variables.

    Key internal state (MATLAB naming)
    ---------------------------------
    ULineMatrix:
        Square matrix updated by sequential Givens rotations (size n_coeffs x n_coeffs).
    dLine_q2:
        Transformed desired vector (size n_coeffs,).
    gamma:
        Likelihood scalar accumulated as a product of cosines in the Givens steps.

    Notes
    -----
    - Real-valued only (supports_complex=False).
    - The returned `errors` correspond to the MATLAB `errorVector`:
        e[k] = d_line * gamma
      and the output is:
        y[k] = d[k] - e[k]
      Therefore we label `error_type="a_posteriori"` to match the MATLAB-style
      “post-rotation” error quantity.
    """

    supports_complex: bool = False

    lamb: float
    n_coeffs: int
    ULineMatrix: np.ndarray
    dLine_q2: np.ndarray
    _tiny: float

    def __init__(
        self,
        filter_order: int,
        lamb: float = 0.99,
        w_init: Optional[ArrayLike] = None,
        *,
        denom_floor: float = 1e-18,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of coefficients is M+1).
        lamb:
            Forgetting factor λ, must satisfy 0 < λ <= 1.
        w_init:
            Optional initial coefficients (length M+1). If None, zeros are used.
        denom_floor:
            Small floor used to avoid division by (near) zero in scalar denominators.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.lamb = float(lamb)
        if not (0.0 < self.lamb <= 1.0):
            raise ValueError(f"lamb must satisfy 0 < lamb <= 1. Got {self.lamb}.")

        self._tiny = float(denom_floor)

        self.n_coeffs = int(self.filter_order) + 1

        self.w = np.asarray(self.w, dtype=np.float64)

        if w_init is not None:
            w0 = np.asarray(w_init, dtype=np.float64).reshape(-1)
            if w0.size != self.n_coeffs:
                raise ValueError(
                    f"w_init must have length {self.n_coeffs}, got {w0.size}."
                )
            self.w = w0.copy()

        self.ULineMatrix = np.zeros((self.n_coeffs, self.n_coeffs), dtype=np.float64)
        self.dLine_q2 = np.zeros(self.n_coeffs, dtype=np.float64)

        self.w_history = []
        self._record_history()

    @staticmethod
    def _givens_rotate_rows(
        row0: np.ndarray,
        row1: np.ndarray,
        cos_t: float,
        sin_t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a 2x2 real Givens rotation:

            [ cos  -sin ] [row0] = [row0']
            [ sin   cos ] [row1]   [row1']

        Parameters
        ----------
        row0, row1:
            1-D arrays (same length) representing stacked rows.
        cos_t, sin_t:
            Givens rotation cosine and sine.

        Returns
        -------
        (row0_rot, row1_rot):
            Rotated rows.
        """
        new0 = cos_t * row0 - sin_t * row1
        new1 = sin_t * row0 + cos_t * row1
        return new0, new1

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
        Run QR-RLS adaptation over (x[k], d[k]) using the MATLAB-style recursion.

        Parameters
        ----------
        input_signal:
            Input sequence x[k] (real), shape (N,).
        desired_signal:
            Desired sequence d[k] (real), shape (N,).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, includes selected internal state in `result.extra`.

        Returns
        -------
        OptimizationResult
            outputs:
                Estimated output y[k] (real).
            errors:
                MATLAB-style error quantity e[k] = d_line * gamma (real).
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_posteriori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["ULineMatrix_last"]:
            Final ULineMatrix.
        extra["dLine_q2_last"]:
            Final dLine_q2.
        extra["gamma_last"]:
            gamma at the last iteration.
        extra["d_line_last"]:
            d_line at the last iteration.
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        n_samples = int(d.size)
        n = int(self.n_coeffs)
        M = int(self.filter_order)

        if n_samples < n:
            raise ValueError(
                f"QR-RLS needs at least (filter_order+1) samples. "
                f"Got n_samples={n_samples}, filter_order={M} => n_coeffs={n}."
            )

        outputs = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        self.ULineMatrix.fill(0.0)
        self.dLine_q2.fill(0.0)

        self.w_history = []
        self._record_history()

        denom0 = float(x[0])
        if abs(denom0) < self._tiny:
            denom0 = self._tiny if denom0 >= 0.0 else -self._tiny

        for kt in range(n):
            w_tmp = np.zeros(n, dtype=np.float64)
            w_tmp[0] = float(d[0] / denom0)

            for ct in range(1, kt + 1):
                num = -float(np.dot(x[1 : ct + 1], w_tmp[ct - 1 :: -1])) + float(d[ct])
                w_tmp[ct] = float(num / denom0)

            self.w = w_tmp
            self._record_history()

            xk = np.zeros(n, dtype=np.float64)
            start = max(0, kt - M)
            seg = x[start : kt + 1][::-1]
            xk[: seg.size] = seg
            outputs[kt] = float(np.dot(w_tmp, xk))

        sqrt_lam = float(np.sqrt(self.lamb))

        for it in range(M + 1):
            scale = float(self.lamb ** ((it + 1) / 2.0))

            vec = x[(n - it - 1) :: -1]
            self.ULineMatrix[it, 0 : (n - it)] = scale * vec

            self.dLine_q2[it] = scale * float(d[n - it - 1])

        gamma_last: float = 1.0
        d_line_last: float = float(d[n - 1])

        for k in range(n, n_samples):
            gamma = 1.0
            d_line = float(d[k])

            reg = x[k : k - M - 1 : -1].copy()

            for rt in range(M + 1):
                row_u = rt
                col_u = n - 1 - rt
                idx_r = n - 1 - rt

                u_val = float(self.ULineMatrix[row_u, col_u])
                r_val = float(reg[idx_r])

                cI = float(np.sqrt(u_val * u_val + r_val * r_val))
                if cI < self._tiny:
                    cos_t, sin_t = 1.0, 0.0
                else:
                    cos_t, sin_t = (u_val / cI), (r_val / cI)

                reg, self.ULineMatrix[row_u, :] = self._givens_rotate_rows(
                    reg, self.ULineMatrix[row_u, :], cos_t, sin_t
                )

                gamma *= cos_t

                dq2_rt = float(self.dLine_q2[row_u])
                new_d_line = (cos_t * d_line) - (sin_t * dq2_rt)
                new_dq2_rt = (sin_t * d_line) + (cos_t * dq2_rt)
                d_line = float(new_d_line)
                self.dLine_q2[row_u] = float(new_dq2_rt)

            d_bar = np.empty(n + 1, dtype=np.float64)
            d_bar[0] = d_line
            d_bar[1:] = self.dLine_q2

            w_new = np.zeros(n, dtype=np.float64)

            den = float(self.ULineMatrix[n - 1, 0])
            if abs(den) < self._tiny:
                den = self._tiny if den >= 0.0 else -self._tiny
            w_new[0] = float(d_bar[n] / den)

            for it in range(1, M + 1):
                row = n - 1 - it
                u_vec = self.ULineMatrix[row, 0:it][::-1]
                w_vec = w_new[0:it][::-1]
                num = -float(np.dot(u_vec, w_vec)) + float(d_bar[n - it])

                den = float(self.ULineMatrix[row, it])
                if abs(den) < self._tiny:
                    den = self._tiny if den >= 0.0 else -self._tiny

                w_new[it] = float(num / den)

            self.w = w_new
            self._record_history()

            self.dLine_q2 *= sqrt_lam
            self.ULineMatrix *= sqrt_lam

            errors[k] = float(d_line * gamma)
            outputs[k] = float(d[k] - errors[k])

            gamma_last = float(gamma)
            d_line_last = float(d_line)

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[QRRLS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "ULineMatrix_last": self.ULineMatrix.copy(),
                "dLine_q2_last": self.dLine_q2.copy(),
                "gamma_last": gamma_last,
                "d_line_last": d_line_last,
                "forgetting_factor": float(self.lamb),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_posteriori",
            extra=extra,
        )
# EOF