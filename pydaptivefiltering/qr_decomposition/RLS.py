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
from pydaptivefiltering._utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class QRRLS(AdaptiveFilter):
    """
    QR-RLS adaptive filter using Givens rotations (real-valued).

    QR-decomposition RLS implementation based on Diniz (Alg. 9.1, 3rd ed.),
    following the reference MATLAB routine ``QR_RLS.m``. This variant maintains
    internal state variables closely matching the MATLAB code and applies
    sequential real Givens rotations to a stacked system.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M+1``.
    lamb : float, optional
        Forgetting factor ``lambda`` with ``0 < lambda <= 1``. Default is 0.99.
    w_init : array_like of float, optional
        Initial coefficient vector ``w(0)`` with shape ``(M+1,)``. If None,
        initializes with zeros.
    denom_floor : float, optional
        Small positive floor used to avoid division by (near) zero in scalar
        denominators. Default is 1e-18.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and coefficients
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    State variables (MATLAB naming)
        This implementation keeps the same key state variables as ``QR_RLS.m``:

        - ``ULineMatrix`` : ndarray, shape ``(M+1, M+1)``
          Upper-triangular-like matrix updated by sequential Givens rotations.
        - ``dLine_q2`` : ndarray, shape ``(M+1,)``
          Transformed desired vector accumulated through the same rotations.
        - ``gamma`` : float
          Scalar accumulated as the product of Givens cosines in each iteration.

    Givens-rotation structure (high level)
        At each iteration, the algorithm applies Givens rotations to eliminate
        components of the stacked vector ``[regressor; ULineMatrix]`` while
        applying the same rotations to ``[d_line; dLine_q2]``. The resulting
        system is then solved by back-substitution to obtain the updated weights.

    Output/error conventions (MATLAB-style)
        The returned ``errors`` correspond to the MATLAB ``errorVector``:

        .. math::
            e[k] = d_{line}[k] \\cdot \\gamma[k],

        and the reported output is computed as:

        .. math::
            y[k] = d[k] - e[k].

        Since this error is formed after the rotation steps (i.e., after the
        QR-update stage), the method sets ``error_type="a_posteriori"``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 3rd ed., Algorithm 9.1 (QR-RLS).
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
        Applies a real 2x2 Givens rotation to a pair of stacked rows.

        The rotation is:

        .. math::
            \\begin{bmatrix}
                \\cos\\theta & -\\sin\\theta \\\\
                \\sin\\theta &  \\cos\\theta
            \\end{bmatrix}
            \\begin{bmatrix}
                \\mathrm{row0} \\\\
                \\mathrm{row1}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
                \\mathrm{row0}' \\\\
                \\mathrm{row1}'
            \\end{bmatrix}.

        Parameters
        ----------
        row0, row1 : ndarray of float
            1-D arrays with the same length (representing two rows to be rotated).
        cos_t, sin_t : float
            Givens rotation cosine and sine.

        Returns
        -------
        (row0_rot, row1_rot) : tuple of ndarray
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
        Executes the QR-RLS adaptation loop (MATLAB-style recursion).

        Parameters
        ----------
        input_signal : array_like of float
            Real-valued input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of float
            Real-valued desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"ULineMatrix_last"``, ``"dLine_q2_last"``, ``"gamma_last"``,
            and ``"d_line_last"``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar output sequence as computed by the MATLAB-style routine:
                ``y[k] = d[k] - e[k]``.
            - errors : ndarray of float, shape ``(N,)``
                MATLAB-style a posteriori error quantity:
                ``e[k] = d_line[k] * gamma[k]``.
            - coefficients : ndarray of float
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_posteriori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``ULineMatrix_last`` : ndarray
                    Final ``ULineMatrix``.
                - ``dLine_q2_last`` : ndarray
                    Final ``dLine_q2``.
                - ``gamma_last`` : float
                    ``gamma`` at the last iteration.
                - ``d_line_last`` : float
                    ``d_line`` at the last iteration.
                - ``forgetting_factor`` : float
                    The forgetting factor ``lambda`` used.
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