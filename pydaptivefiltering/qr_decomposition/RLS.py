# pydaptivefiltering/QR/RLS.py
#
#       Implements the QR-RLS algorithm for REAL valued data.
#       (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, 3rd Ed., Diniz)
#
#       Notes:
#       - This implementation follows the reference MATLAB code you provided:
#         QR_RLS.m (REAL-valued) with the same state variables:
#           ULineMatrix, dLine_q2, regressorLine, dLine, gamma, and the same
#         “Givens rotation on stacked [reg; U] and [dLine; dLine_q2]”.
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                               diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Optional, Union, List, Dict

from pydaptivefiltering.base import AdaptiveFilter


class QRRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the QR-RLS algorithm for REAL valued data.
        (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical
         Implementation, 3rd Ed., Diniz)

        This implementation is intentionally REAL-only (supports_complex=False)
        and mirrors the MATLAB reference routine provided by the authors.

    Key State Variables (as in MATLAB)
    ---------------------------------
        ULineMatrix : (nCoeff x nCoeff) upper-triangular-like matrix updated by Givens rotations.
        dLine_q2    : (nCoeff,) transformed desired vector.
        gamma       : likelihood scalar accumulated as product of cosines.
    """

    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int,
        lamb: float = 0.99,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                FIR order (M). Number of coefficients is M+1.
            lamb : float
                Forgetting factor lambda, 0 < lambda < 1 (or <= 1 if desired).
            w_init : array_like, optional
                Initial coefficients (length M+1). If None, zeros are used.

        Notes
        -----
            Internally we maintain ULineMatrix and dLine_q2 exactly like the MATLAB code.
        """
        super().__init__(filter_order, w_init)

        self.lamb = float(lamb)
        if not (0.0 < self.lamb < 1.0 or np.isclose(self.lamb, 1.0)):
            raise ValueError(f"lamb must satisfy 0 < lamb <= 1. Got {self.lamb}.")

        self.n_coeffs: int = self.m + 1

        # MATLAB states
        self.ULineMatrix: np.ndarray = np.zeros((self.n_coeffs, self.n_coeffs), dtype=float)
        self.dLine_q2: np.ndarray = np.zeros(self.n_coeffs, dtype=float)

    @staticmethod
    def _ensure_real(x: np.ndarray, name: str) -> np.ndarray:
        """Reject complex arrays (REAL-only algorithm)."""
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise TypeError(f"{name} must be REAL. This QR_RLS implementation is real-only.")
        return x.astype(float, copy=False)

    def _givens_rotate_rows(
        self,
        row0: np.ndarray,
        row1: np.ndarray,
        cos_t: float,
        sin_t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the 2x2 real Givens rotation (as in the MATLAB QLineTeta blocks):

            [ cos  -sin ] [row0] = [row0']
            [ sin   cos ] [row1]   [row1']

        Returns updated (row0', row1').
        """
        new0 = cos_t * row0 - sin_t * row1
        new1 = sin_t * row0 + cos_t * row1
        return new0, new1

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the REAL QR-RLS algorithm
            using the same steps as the provided MATLAB implementation.

        Inputs
        -------
            input_signal   : np.ndarray | list
                Input signal (x). 1-D array.
            desired_signal : np.ndarray | list
                Desired signal (d). 1-D array, same length as input_signal.
            verbose        : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs      : Estimated output per iteration (REAL).
                errors       : A POSTERIORI error per iteration (REAL)  [matches MATLAB errorVector].
                coefficients : Coefficients history (list of vectors).

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz     -                               diniz@lps.ufrj.br
        """
        tic = time()

        x = self._ensure_real(np.asarray(input_signal), "input_signal")
        d = self._ensure_real(np.asarray(desired_signal), "desired_signal")
        self._validate_inputs(x, d)

        n_iterations = d.size
        n = self.n_coeffs
        M = self.m

        # Need at least n samples to build initial matrices exactly as MATLAB
        if n_iterations < n:
            raise ValueError(
                f"QR_RLS needs at least (filter_order+1) samples. "
                f"Got n_samples={n_iterations}, filter_order={M} => n_coeffs={n}."
            )

        # Pre-allocations like MATLAB
        y = np.zeros(n_iterations, dtype=float)
        error_vec = np.zeros(n_iterations, dtype=float)

        # ============================================================
        # Backsubstitution Procedure (MATLAB initialization of coefficients)
        # ============================================================
        # MATLAB uses coefficientVector(:,kt) for kt=1..n
        # We will build w_history up to n-1 using the same recursion, but store
        # full length-n vectors (pad zeros for not-yet-defined coefficients).
        #
        # WARNING: MATLAB assumes input(1) != 0; we guard with tiny epsilon.
        eps = 1e-18
        denom0 = x[0] if abs(x[0]) > eps else (eps if x[0] >= 0 else -eps)

        # reset our history to match what we build here
        self.w_history = []
        self.w = np.zeros(n, dtype=complex)  # keep base class shape; we use real values

        # kt=1..n (1-based) => kt_idx=0..n-1 (0-based)
        for kt_idx in range(n):
            # coefficients for order kt_idx (meaning length kt_idx+1)
            w_tmp = np.zeros(n, dtype=float)

            # coefficientVector(1,kt) = desired(1)/input(1)
            w_tmp[0] = d[0] / denom0

            # for ct = 2:kt
            # ( -input(2:ct)*coefficientVector(ct-1:-1:1,kt) + desired(ct))/input(1)
            # In 0-based: ct_idx from 1..kt_idx
            for ct_idx in range(1, kt_idx + 1):
                # input(2:ct) in MATLAB => x[1:ct_idx+1] in python
                # coefficientVector(ct-1:-1:1) => w_tmp[ct_idx-1::-1]
                num = -np.dot(x[1: ct_idx + 1], w_tmp[ct_idx - 1 :: -1]) + d[ct_idx]
                w_tmp[ct_idx] = num / denom0

            # store and set current w
            self.w = w_tmp.astype(float).astype(complex)
            self.w_history.append(self.w.copy())

            # output estimate using current coefficients (consistent FIR TDL)
            # For k=kt_idx we can form regressor x[k::-1] padded by zeros
            xk = np.zeros(n, dtype=float)
            start = max(0, kt_idx - M)
            seg = x[start : kt_idx + 1][::-1]
            xk[: seg.size] = seg
            y[kt_idx] = float(np.dot(w_tmp, xk))
            # error_vec early positions are not defined by MATLAB loop, keep 0.0

        # ============================================================
        # Build Initial Matrices (ULineMatrix and dLine_q2) — MATLAB style
        # ============================================================
        # MATLAB:
        # for it = 0:M
        #   ULineMatrix(it+1,1:end-it) = (lambda^((it+1)/2))*input(n-it:-1:1);
        #   dLine_q2(it+1)             = (lambda^((it+1)/2))*desired(n-it);
        #
        # Here input and desired are row vectors in MATLAB 1-based indexing.
        self.ULineMatrix[:] = 0.0
        self.dLine_q2[:] = 0.0

        sqrt_lam = np.sqrt(self.lamb)

        for it in range(0, M + 1):
            scale = self.lamb ** ((it + 1) / 2.0)

            # input(n-it:-1:1) => x[n-1-it : -1 : -1] then reversed length (n-it)
            # In python: x[n-it-1::-1] gives x[n-it-1],...,x[0] (length n-it)
            vec = x[(n - it - 1) :: -1]  # length n-it

            # place into columns 0 .. (n-it-1)
            self.ULineMatrix[it, 0 : (n - it)] = scale * vec

            # desired(n-it) => d[n-it-1]
            self.dLine_q2[it] = scale * d[n - it - 1]

        # ============================================================
        # Main Loop — MATLAB: for kt = nCoefficients+1 : nIterations
        # ============================================================
        # 0-based: k = n .. n_iterations-1
        for k in range(n, n_iterations):
            gamma = 1.0
            d_line = float(d[k])

            # regressorLine = input(kt:-1:kt-M)  (length n)
            reg = x[k : k - M - 1 : -1].copy()  # x[k], x[k-1], ... x[k-M]

            # Givens Rotations for rt = 0:M
            for rt in range(0, M + 1):
                # MATLAB indices:
                # ULineMatrix(rt+1, n-rt)  and  regressorLine(end-rt)
                # 0-based:
                row_u = rt
                col_u = n - 1 - rt
                idx_r = n - 1 - rt

                u_val = self.ULineMatrix[row_u, col_u]
                r_val = reg[idx_r]

                cI = np.sqrt(u_val * u_val + r_val * r_val)
                if cI < eps:
                    cos_t = 1.0
                    sin_t = 0.0
                else:
                    cos_t = u_val / cI
                    sin_t = r_val / cI

                # Apply rotation to stacked [reg; ULineMatrix] affecting:
                # - row 0   : reg
                # - row rt+1: ULineMatrix[rt, :]
                reg, self.ULineMatrix[row_u, :] = self._givens_rotate_rows(
                    reg, self.ULineMatrix[row_u, :], cos_t, sin_t
                )

                gamma *= cos_t

                # Apply same rotation to stacked [d_line; dLine_q2] affecting:
                # - element 0: d_line
                # - element rt+1: dLine_q2[rt]
                # new_d_line = cos*d_line - sin*dLine_q2[rt]
                # new_dq2_rt = sin*d_line + cos*dLine_q2[rt]
                dq2_rt = self.dLine_q2[row_u]
                new_d_line = cos_t * d_line - sin_t * dq2_rt
                new_dq2_rt = sin_t * d_line + cos_t * dq2_rt
                d_line = new_d_line
                self.dLine_q2[row_u] = new_dq2_rt

            # Compute Coefficient Vector (MATLAB dBar = [dLine ; dLine_q2])
            d_bar = np.empty(n + 1, dtype=float)
            d_bar[0] = d_line
            d_bar[1:] = self.dLine_q2

            # coefficientVector(1,kt) = dBar(n+1)/ULineMatrix(n,1)
            # 0-based: w[0] = d_bar[n] / U[n-1,0]
            w_new = np.zeros(n, dtype=float)

            den = self.ULineMatrix[n - 1, 0]
            if abs(den) < eps:
                den = eps if den >= 0 else -eps
            w_new[0] = d_bar[n] / den

            # for it = 1:M
            #   w(it+1) = ( -U(n-it, it:-1:1)*w(it:-1:1) + dBar(n+1-it) ) / U(n-it, 1+it)
            for it in range(1, M + 1):
                row = n - 1 - it
                # columns 0..it-1 reversed (matches it:-1:1 in MATLAB)
                u_vec = self.ULineMatrix[row, 0:it][::-1]
                w_vec = w_new[0:it][::-1]
                num = -np.dot(u_vec, w_vec) + d_bar[n - it]
                den = self.ULineMatrix[row, it]
                if abs(den) < eps:
                    den = eps if den >= 0 else -eps
                w_new[it] = num / den

            # Update filter weights and history
            self.w = w_new.astype(complex)
            self.w_history.append(self.w.copy())

            # Update Parameters (MATLAB scales at the end of iteration)
            self.dLine_q2 *= sqrt_lam
            self.ULineMatrix *= sqrt_lam

            # MATLAB: errorVector(kt) = dLine * gamma  (A POSTERIORI error)
            error_vec[k] = d_line * gamma

            # Provide an output estimate for completeness.
            # We choose y[k] = d[k] - error_vec[k] so shapes are consistent and REAL.
            y[k] = d[k] - error_vec[k]

        if verbose:
            print(f"[QR-RLS REAL] Completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": y,
            "errors": error_vec,          
            "coefficients": self.w_history,
        }


# EOF
