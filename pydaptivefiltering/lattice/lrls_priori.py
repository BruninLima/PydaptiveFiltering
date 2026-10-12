# lattice.lrls_priori.py
#
#      Implements the Lattice RLS algorithm based on a priori errors.
#      (Algorithm 7.4 - book: Adaptive Filtering: Algorithms and Practical
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


class LRLSPriori(AdaptiveFilter):
    """
    Lattice RLS using a priori errors (LRLS, a priori form), complex-valued.

    Implements Diniz (Algorithm 7.4) in a lattice/ladder structure:

    1) **Lattice prediction stage** (order ``M``):
       Produces forward a priori errors and a vector of backward errors, updating
       reflection-like state variables and exponentially weighted energies.

    2) **Ladder (joint-process) stage** (length ``M+1``):
       Updates the ladder coefficients ``v`` using the a priori backward-error
       vector and produces an **a priori** error associated with the desired signal.

    Library conventions
    -------------------
    - Complex-valued implementation (``supports_complex=True``).
    - Ladder coefficients are stored in ``self.v`` with length ``M+1``.
    - For compatibility with :class:`~pydaptivefiltering.base.AdaptiveFilter`,
      ``self.w`` mirrors ``self.v`` at each iteration and the base-class history
      corresponds to the ladder coefficient trajectory.

    Parameters
    ----------
    filter_order : int
        Lattice order ``M`` (number of sections). The ladder has ``M+1`` coefficients.
    lambda_factor : float, optional
        Forgetting factor ``lambda`` used in the exponentially weighted recursions.
        Default is 0.99.
    epsilon : float, optional
        Initialization/regularization constant for the energy variables
        (forward/backward). Default is 0.1.
    w_init : array_like of complex, optional
        Optional initial ladder coefficients of length ``M+1``. If None, initializes
        with zeros.
    denom_floor : float, optional
        Small positive floor used to avoid division by (near) zero in normalization
        terms (``gamma`` variables and energy denominators). Default is 1e-12.

    Notes
    -----
    Signals and dimensions
    ~~~~~~~~~~~~~~~~~~~~~~
    For lattice order ``M``:

    - ``delta`` has shape ``(M,)`` (lattice delta state)
    - ``xi_f`` and ``xi_b`` have shape ``(M+1,)`` (forward/backward energies)
    - ``error_b_prev`` and per-sample ``alpha_b`` have shape ``(M+1,)``
      (backward-error vectors)
    - ``v`` and ``delta_v`` have shape ``(M+1,)`` (ladder coefficients and state)

    A priori error (as returned)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The ladder stage starts with ``alpha_e = d[k]`` and removes components
    correlated with the backward-error vector:

    .. math::
        \\alpha_e \\leftarrow \\alpha_e - v_m^*(k-1)\\, b_m(k),

    where :math:`b_m(k)` are the a priori backward errors (``alpha_b[m]``).
    The final value is the a priori error ``e_pri(k)``, returned in
    ``errors[k]``; the output estimate is ``outputs[k] = d[k] - e_pri[k]``.
    (The a posteriori error is ``gamma_{M+1}(k) * e_pri(k)``.)

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 7.4.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Number of lattice sections M. Ladder has M+1 coefficients.
        lambda_factor:
            Forgetting factor λ.
        epsilon:
            Energy initialization / regularization.
        w_init:
            Optional initial ladder coefficient vector (length M+1). If None, zeros.
        denom_floor:
            Floor used to avoid division by (near) zero in normalization terms.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)
        self._tiny = float(denom_floor)

        self.delta = np.zeros(self.n_sections, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)
        self.gamma = np.ones(self.n_sections + 1, dtype=float)
        self.kappa_f = np.zeros(self.n_sections, dtype=complex)
        self.kappa_b = np.zeros(self.n_sections, dtype=complex)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

        # Mirror to base API
        self.w = self.v.copy()
        self.w_history = []
        self._record_history()

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes LRLS adaptation (a priori form) over paired sequences ``x[k]`` and ``d[k]``.

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
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Estimated output sequence. In this implementation:
                ``outputs[k] = d[k] - e_pri[k]``.
            - errors : ndarray of complex, shape ``(N,)``
                A priori ladder error ``e_pri[k]``.
            - coefficients : ndarray
                Ladder coefficient history (mirrors ``self.v`` via ``self.w``).
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` (see below).

        Extra (when return_internal_states=True)
        --------------------------------------
        xi_f : ndarray of float, shape ``(M+1,)``
            Final forward energies.
        xi_b : ndarray of float, shape ``(M+1,)``
            Final backward energies.
        delta : ndarray of complex, shape ``(M,)``
            Final lattice delta state.
        delta_v : ndarray of complex, shape ``(M+1,)``
            Final ladder delta state used to compute ``v``.
        """
        t0 = perf_counter()

        # validate_input already normalizes to 1D and matches lengths.
        # Force complex to respect supports_complex=True (even if x/d are real).
        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            # Values from time k-1 needed by the order recursions.
            xi_b_prev = self.xi_b.copy()
            gamma_prev = self.gamma.copy()

            alpha_f = complex(x_in[k])

            alpha_b = np.zeros(self.n_sections + 1, dtype=complex)
            alpha_b[0] = x_in[k]

            gamma = np.ones(self.n_sections + 1, dtype=float)

            energy_x = float(np.real(alpha_f * np.conj(alpha_f)))
            self.xi_f[0] = self.lam * self.xi_f[0] + energy_x
            self.xi_b[0] = self.xi_f[0]

            # -------------------------
            # Lattice stage (a priori)
            # -------------------------
            for m in range(self.n_sections):
                # A priori errors use the reflection coefficients from time k-1.
                alpha_f_next = alpha_f - self.kappa_f[m] * self.error_b_prev[m]
                alpha_b[m + 1] = self.error_b_prev[m] - self.kappa_b[m] * alpha_f

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + gamma_prev[m] * self.error_b_prev[m] * np.conj(alpha_f)
                )

                energy_b = float(np.real(alpha_b[m] * np.conj(alpha_b[m])))
                gamma[m + 1] = max(
                    gamma[m] - gamma[m] ** 2 * energy_b / (self.xi_b[m] + self._tiny),
                    self._tiny,
                )

                self.kappa_b[m] = self.delta[m] / (self.xi_f[m] + self._tiny)
                self.kappa_f[m] = np.conj(self.delta[m]) / (xi_b_prev[m] + self._tiny)

                delta_sq = float(np.real(self.delta[m] * np.conj(self.delta[m])))
                self.xi_b[m + 1] = max(xi_b_prev[m] - delta_sq / (self.xi_f[m] + self._tiny), self._tiny)
                self.xi_f[m + 1] = max(self.xi_f[m] - delta_sq / (xi_b_prev[m] + self._tiny), self._tiny)

                alpha_f = alpha_f_next

            # -------------------------
            # Ladder stage (a priori)
            # -------------------------
            alpha_e = complex(d_in[k])

            for m in range(self.n_sections + 1):
                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + gamma[m] * alpha_b[m] * np.conj(alpha_e)
                )

                # A priori error uses the ladder coefficient from time k-1.
                alpha_e = alpha_e - np.conj(self.v[m]) * alpha_b[m]

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)

            self.gamma = gamma

            e_k = alpha_e
            errors[k] = e_k
            outputs[k] = d_in[k] - e_k

            self.error_b_prev = alpha_b.copy()

            # Mirror ladder coeffs into base API + record history
            self.w = self.v.copy()
            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[LRLSPriori] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "xi_f": self.xi_f.copy(),
                "xi_b": self.xi_b.copy(),
                "delta": self.delta.copy(),
                "delta_v": self.delta_v.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF