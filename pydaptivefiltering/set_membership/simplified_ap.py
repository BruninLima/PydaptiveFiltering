#  set_membership.simplified_ap.py
#
#       Implements the Simplified Set-membership Affine-Projection (SM-Simp-AP)
#       algorithm for COMPLEX valued data.
#       (Algorithm 6.3 - book: Adaptive Filtering: Algorithms and Practical
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

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SimplifiedSMAP(AdaptiveFilter):
    """
    Simplified Set-Membership Affine Projection (SM-Simp-AP) adaptive filter
    (complex-valued).

    Implements Algorithm 6.3 (Diniz). This is a simplified affine-projection
    set-membership scheme where an AP-style regressor matrix of length ``L+1``
    is maintained, but **the update uses only the most recent column** (the
    current regressor vector). Updates occur only when the a priori error
    magnitude exceeds ``gamma_bar``.

    Parameters
    ----------
    filter_order : int
        FIR filter order ``M`` (number of coefficients is ``M + 1``).
    gamma_bar : float
        Set-membership bound ``\\bar{\\gamma}`` for the a priori error magnitude.
        An update occurs only if ``|e[k]| > gamma_bar``.
    gamma : float
        Regularization constant used in the normalization denominator
        ``gamma + ||x_k||^2``.
    L : int
        Reuse data factor / constraint length. In this simplified variant it
        mainly determines the number of columns kept in the internal AP-style
        regressor matrix (size ``(M+1) x (L+1)``); only the first column is used
        in the update.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)``, shape ``(M + 1,)``. If None, zeros.

    Notes
    -----
    Regressor definition
        The current tapped-delay regressor is

        .. math::
            x_k = [x[k], x[k-1], \\dots, x[k-M]]^T \\in \\mathbb{C}^{M+1}.

        Internally, the algorithm maintains an AP regressor matrix

        .. math::
            X_k = [x_k, x_{k-1}, \\dots, x_{k-L}] \\in \\mathbb{C}^{(M+1)\\times(L+1)},

        but the update uses only the first column ``x_k``.

    A priori output and error (as implemented)
        This implementation computes

        .. math::
            y[k] = x_k^H w[k],

        and stores it as ``outputs[k]``.
        The stored error is

        .. math::
            e[k] = d^*[k] - y[k].

        (This matches the semantics of your code; many texts use
        ``e[k] = d[k] - w^H x_k``. If you want the textbook convention, you’d
        remove the conjugation on ``d[k]`` and ensure ``y[k]=w^H x_k``.)

    Set-membership condition
        If ``|e[k]| \\le \\bar{\\gamma}``, no update is performed.

        If ``|e[k]| > \\bar{\\gamma}``, define the scalar step factor

        .. math::
            s[k] = \\left(1 - \\frac{\\bar{\\gamma}}{|e[k]|}\\right) e[k].

    Normalized update (simplified AP)
        With ``\\mathrm{den}[k] = \\gamma + \\|x_k\\|^2``, the coefficient update is

        .. math::
            w[k+1] = w[k] + \\frac{s[k]}{\\mathrm{den}[k]} \\, x_k.

    Returned error type
        The returned sequences correspond to **a priori** quantities (computed
        before updating ``w``), so ``error_type="a_priori"``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 6.3.
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
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)
        self.n_coeffs = int(self.filter_order + 1)

        self.regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)

        self.X_matrix = self.regressor_matrix

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
        Executes the SM-Simp-AP adaptation.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]``, shape ``(N,)`` (flattened internally).
        desired_signal : array_like of complex
            Desired sequence ``d[k]``, shape ``(N,)`` (flattened internally).
        verbose : bool, optional
            If True, prints runtime and update statistics after completion.
        return_internal_states : bool, optional
            If True, includes internal trajectories in ``result.extra``:
            ``step_factor`` and ``den`` (each length ``N``). Entries are zero
            when no update occurs.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                A priori output sequence.
            - errors : ndarray of complex, shape ``(N,)``
                A priori error sequence (as in code: ``e[k] = conj(d[k]) - y[k]``).
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict
                Always present with:
                - ``"n_updates"`` : int
                    Number of coefficient updates (iterations where ``|e[k]| > gamma_bar``).
                - ``"update_mask"`` : ndarray of bool, shape ``(N,)``
                    Boolean mask indicating which iterations performed updates.
                Additionally present only if ``return_internal_states=True``:
                - ``"step_factor"`` : ndarray of complex, shape ``(N,)``
                    Scalar factor ``(1 - gamma_bar/|e|) * e`` (0 when no update).
                - ``"den"`` : ndarray of float, shape ``(N,)``
                    Denominator ``gamma + ||x_k||^2`` (0 when no update).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        step_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None
        den_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])

        for k in range(n_samples):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]

            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            xk: np.ndarray = self.regressor_matrix[:, 0:1]

            output_k: complex = complex((xk.conj().T @ w_current).item())
            error_k: complex = complex(np.conj(d[k]) - output_k)

            outputs[k] = output_k
            errors[k] = error_k

            eabs: float = float(np.abs(error_k))

            if eabs > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True

                step_factor: complex = complex((1.0 - (self.gamma_bar / eabs)) * error_k)

                norm_sq: float = float(np.real((xk.conj().T @ xk).item()))
                den: float = float(self.gamma + norm_sq)
                if den <= 0.0:
                    den = float(self.gamma + 1e-30)

                w_current = w_current + (step_factor / den) * xk

                if return_internal_states:
                    if step_track is not None:
                        step_track[k] = step_factor
                    if den_track is not None:
                        den_track[k] = den
            else:
                if return_internal_states:
                    if step_track is not None:
                        step_track[k] = 0.0 + 0.0j
                    if den_track is not None:
                        den_track[k] = 0.0

            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-Simp-AP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.2f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "step_factor": step_track,
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