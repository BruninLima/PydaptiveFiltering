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
    Set-Membership Normalized LMS (SM-NLMS) adaptive filter (complex-valued).

    Implements Algorithm 6.1 (Diniz). The coefficients are updated **only** when
    the magnitude of the a priori error exceeds a prescribed bound ``gamma_bar``
    (set-membership criterion). When an update occurs, a normalized LMS-like
    step is applied with an effective step factor that depends on ``|e[k]|``.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M`` (number of coefficients is ``M + 1``).
    gamma_bar : float
        Set-membership bound ``\\bar{\\gamma}`` for the a priori error magnitude.
        An update occurs only if ``|e[k]| > gamma_bar``.
    gamma : float
        Regularization constant used in the NLMS denominator
        ``gamma + ||x_k||^2`` to improve numerical stability.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)``, shape ``(M + 1,)``. If None, zeros.

    Notes
    -----
    Let the tapped-delay regressor be

    .. math::
        x_k = [x[k], x[k-1], \\dots, x[k-M]]^T \\in \\mathbb{C}^{M+1}.

    The a priori output and error are

    .. math::
        y[k] = w^H[k] x_k, \\qquad e[k] = d[k] - y[k].

    Set-membership condition
        If ``|e[k]| \\le \\bar{\\gamma}``, no update is performed.

        If ``|e[k]| > \\bar{\\gamma}``, define the SM step factor

        .. math::
            \\mu[k] = 1 - \\frac{\\bar{\\gamma}}{|e[k]|} \\in (0,1).

    Normalized update (as implemented)
        With ``\\mathrm{den}[k] = \\gamma + \\|x_k\\|^2``, the coefficient update is

        .. math::
            w[k+1] = w[k] + \\frac{\\mu[k]}{\\mathrm{den}[k]} \\, e^*[k] \\, x_k.

    Returned error type
        This implementation reports the **a priori** sequences (computed before
        updating ``w``), so ``error_type="a_priori"``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 6.1.
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
        Executes the SM-NLMS adaptation over paired sequences ``x[k]`` and ``d[k]``.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (flattened internally).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (flattened internally).
        verbose : bool, optional
            If True, prints runtime and update statistics after completion.
        return_internal_states : bool, optional
            If True, includes internal trajectories in ``result.extra``:
            ``mu`` and ``den`` (each length ``N``). Entries are zero when no update occurs.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                A priori output sequence ``y[k] = w^H[k] x_k``.
            - errors : ndarray of complex, shape ``(N,)``
                A priori error sequence ``e[k] = d[k] - y[k]``.
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
                - ``"mu"`` : ndarray of float, shape ``(N,)``
                    Step factor ``mu[k]`` (0 when no update).
                - ``"den"`` : ndarray of float, shape ``(N,)``
                    Denominator ``gamma + ||x_k||^2`` (0 when no update).
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