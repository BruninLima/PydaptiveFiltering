# lattice.normalized_lrls.py
#
#      Implements the Normalized Lattice RLS algorithm based on a posteriori error.
#      (Algorithm 7.6 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import perf_counter
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class NormalizedLRLS(AdaptiveFilter):
    """
    Normalized Lattice RLS (NLRLS) based on a posteriori error, complex-valued.

    Implements Diniz (Algorithm 7.6). This variant introduces *normalized*
    internal variables so that key quantities (normalized forward/backward errors
    and reflection-like coefficients) are designed to be magnitude-bounded by 1,
    improving numerical robustness.

    The algorithm has two coupled stages:

    1) **Prediction stage (lattice, order M)**:
       Computes normalized forward/backward a posteriori errors (``bar_f``, ``bar_b``)
       and updates normalized reflection-like coefficients ``rho``.

    2) **Estimation stage (normalized ladder, length M+1)**:
       Updates normalized coefficients ``rho_v`` and produces a normalized
       a posteriori error ``bar_e``. The returned error is the *de-normalized*
       error ``e = bar_e * xi_half``.

    Library conventions
    -------------------
    - Complex-valued implementation (``supports_complex=True``).
    - The exposed coefficient vector is ``rho_v`` (length ``M+1``).
      For compatibility with :class:`~pydaptivefiltering.base.AdaptiveFilter`:
        * ``self.w`` mirrors ``self.rho_v`` at each iteration.
        * history recorded by ``_record_history()`` corresponds to ``rho_v``.

    Parameters
    ----------
    filter_order : int
        Lattice order ``M`` (number of sections). The estimation stage uses
        ``M+1`` coefficients.
    lambda_factor : float, optional
        Forgetting factor ``lambda`` used in the exponentially weighted updates.
        Default is 0.99.
    epsilon : float, optional
        Small positive constant used for regularization in normalizations,
        magnitude clipping, and denominator protection. Default is 1e-6.
    w_init : array_like of complex, optional
        Optional initialization for ``rho_v`` with length ``M+1``.
        If None, initializes with zeros.
    denom_floor : float, optional
        Extra floor for denominators and sqrt protections. Default is 1e-12.

    Notes
    -----
    Normalized variables
    ~~~~~~~~~~~~~~~~~~~~
    The implementation uses the following normalized quantities:

    - ``xi_half``: square-root energy tracker (scalar). It normalizes the input/output
      so that normalized errors stay bounded.
    - ``bar_f``: normalized forward error for the current section.
    - ``bar_b_prev`` / ``bar_b_curr``: normalized backward error vectors, shape ``(M+1,)``.
    - ``bar_e``: normalized a posteriori error in the estimation stage.
    - ``rho``: normalized reflection-like coefficients for the lattice stage, shape ``(M,)``.
    - ``rho_v``: normalized coefficients for the estimation stage, shape ``(M+1,)``.

    Magnitude bounding
    ~~~~~~~~~~~~~~~~~~
    Several variables are clipped to satisfy ``|z| <= 1``. The terms
    ``sqrt(1 - |z|^2)`` act like cosine factors in the normalized recursions and
    are safeguarded with ``_safe_sqrt`` to avoid negative arguments caused by
    round-off.

    Output and error returned
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    The filter returns the de-normalized a posteriori error:

    ``errors[k] = bar_e[k] * xi_half[k]``

    and the output estimate:

    ``outputs[k] = d[k] - errors[k]``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 7.6.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Number of lattice sections M. The estimation stage uses M+1 coefficients.
        lambda_factor:
            Forgetting factor λ.
        epsilon:
            Regularization used in normalizations and clipping.
        w_init:
            Optional initialization for rho_v (length M+1). If None, zeros.
        denom_floor:
            Extra floor for denominators / sqrt protections.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)
        self._tiny = float(denom_floor)

        self.rho = np.zeros(self.n_sections, dtype=complex)

        if w_init is not None:
            rho_v0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if rho_v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {rho_v0.size}"
                )
            self.rho_v = rho_v0
        else:
            self.rho_v = np.zeros(self.n_sections + 1, dtype=complex)

        self.bar_b_prev = np.zeros(self.n_sections + 1, dtype=complex)
        self.xi_half = float(np.sqrt(self.epsilon))

        self.w = self.rho_v.copy()
        self.w_history = []
        self._record_history()

    @staticmethod
    def _safe_sqrt(value: float) -> float:
        """
        Computes sqrt(max(value, 0.0)) to avoid negative arguments due to rounding.
        """
        return float(np.sqrt(max(0.0, float(value))))

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run the Normalized LRLS (NLRLS) recursion over paired sequences ``x[k]`` and ``d[k]``.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)``.
        desired_signal : array_like of complex
            Desired/reference sequence ``d[k]`` with shape ``(N,)``.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, returns selected *final* internal states in ``result.extra``
            (not full trajectories).

        Returns
        -------
        OptimizationResult
            outputs : ndarray of complex, shape ``(N,)``
                Estimated output sequence ``y[k] = d[k] - e_post[k]``.
            errors : ndarray of complex, shape ``(N,)``
                De-normalized a posteriori error ``e_post[k] = bar_e[k] * xi_half[k]``.
            coefficients : ndarray
                History of ``rho_v`` (mirrors ``self.rho_v`` via ``self.w``).
            error_type : str
                Set to ``"a_posteriori"``.
            extra : dict, optional
                Present only if ``return_internal_states=True`` (see below).

        Extra (when return_internal_states=True)
        --------------------------------------
        rho : ndarray of complex, shape ``(M,)``
            Final normalized lattice reflection-like coefficients.
        rho_v : ndarray of complex, shape ``(M+1,)``
            Final normalized estimation-stage coefficients.
        xi_half : float
            Final square-root energy tracker used for normalization.
        """
        t0 = perf_counter()

        # validate_input already normalizes to 1D and matches lengths.
        # Force complex to respect supports_complex=True (even if x/d are real).
        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        sqrt_lam = float(np.sqrt(self.lam))

        for k in range(n_samples):
            # Update xi_half (sqrt energy)
            xi_sq = float(self.xi_half**2)
            xi_sq = float(self.lam * xi_sq + (np.abs(x_in[k]) ** 2))
            self.xi_half = self._safe_sqrt(xi_sq)

            denom_x = float(self.xi_half + self.epsilon)
            bar_f = x_in[k] / denom_x

            abs_bf = np.abs(bar_f)
            if abs_bf > 1.0:
                bar_f = bar_f / abs_bf

            bar_b_curr = np.zeros(self.n_sections + 1, dtype=complex)
            bar_b_curr[0] = bar_f

            # -------------------------
            # Prediction stage
            # -------------------------
            for m in range(self.n_sections):
                cos_f = self._safe_sqrt(1.0 - (np.abs(bar_f) ** 2))
                cos_b_prev = self._safe_sqrt(1.0 - (np.abs(self.bar_b_prev[m]) ** 2))

                self.rho[m] = (
                    (sqrt_lam * cos_f * cos_b_prev * self.rho[m])
                    + (np.conj(bar_f) * self.bar_b_prev[m])
                )

                abs_rho = np.abs(self.rho[m])
                if abs_rho >= 1.0:
                    self.rho[m] = self.rho[m] / (abs_rho + self.epsilon)

                cos_rho = self._safe_sqrt(1.0 - (np.abs(self.rho[m]) ** 2))

                denom_f = float((cos_rho * cos_b_prev) + self.epsilon)
                denom_b = float((cos_rho * cos_f) + self.epsilon)

                f_next = (bar_f - self.rho[m] * self.bar_b_prev[m]) / denom_f
                b_next = (self.bar_b_prev[m] - np.conj(self.rho[m]) * bar_f) / denom_b

                bar_f = f_next
                bar_b_curr[m + 1] = b_next

            # -------------------------
            # Estimation stage
            # -------------------------
            bar_e = d_in[k] / float(self.xi_half + self.epsilon)
            abs_be = np.abs(bar_e)
            if abs_be > 1.0:
                bar_e = bar_e / abs_be

            for m in range(self.n_sections + 1):
                cos_e = self._safe_sqrt(1.0 - (np.abs(bar_e) ** 2))
                cos_b = self._safe_sqrt(1.0 - (np.abs(bar_b_curr[m]) ** 2))

                self.rho_v[m] = (
                    (sqrt_lam * cos_e * cos_b * self.rho_v[m])
                    + (np.conj(bar_e) * bar_b_curr[m])
                )

                abs_rv = np.abs(self.rho_v[m])
                if abs_rv >= 1.0:
                    self.rho_v[m] = self.rho_v[m] / (abs_rv + self.epsilon)

                cos_rho_v = self._safe_sqrt(1.0 - (np.abs(self.rho_v[m]) ** 2))

                denom_e = float((cos_rho_v * cos_b) + self.epsilon)
                bar_e = (bar_e - self.rho_v[m] * bar_b_curr[m]) / denom_e

            errors[k] = bar_e * self.xi_half
            outputs[k] = d_in[k] - errors[k]

            self.bar_b_prev = bar_b_curr.copy()

            self.w = self.rho_v.copy()
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[NormalizedLRLS] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "rho": self.rho.copy(),
                "rho_v": self.rho_v.copy(),
                "xi_half": self.xi_half,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_posteriori",
            extra=extra,
        )
# EOF