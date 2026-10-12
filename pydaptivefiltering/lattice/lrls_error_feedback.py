# lattice.lrls_error_feedback.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors
#      with Error Feedback.
#      (Algorithm 7.5 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@wam@gmail.com
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering._utils.typing import ComplexArrayLike, ArrayLike

class LRLSErrorFeedback(AdaptiveFilter):
    """
    Lattice RLS with a posteriori errors and Error Feedback (LRLS-EF), complex-valued.

    Implements the lattice/ladder RLS structure with error feedback described in
    Diniz (Algorithm 7.5). The method decomposes the adaptation into:

    1) **Lattice prediction stage**:
       Updates forward/backward a posteriori prediction errors and associated
       reflection-like variables via exponentially weighted energies.

    2) **Ladder (joint-process) stage**:
       Estimates the ladder coefficients that map the lattice backward-error
       vector into the desired response.

    In this implementation, the ladder coefficient vector is stored in ``self.v``
    (length ``M+1``). For compatibility with :class:`~pydaptivefiltering.base.AdaptiveFilter`,
    ``self.w`` mirrors ``self.v`` at each iteration and the coefficient history
    recorded by the base class corresponds to the ladder coefficients.

    Parameters
    ----------
    filter_order : int
        Lattice order ``M`` (number of sections). The ladder has ``M+1`` coefficients.
    lambda_factor : float, optional
        Forgetting factor ``lambda`` used in the exponentially weighted recursions.
        Default is 0.99.
    epsilon : float, optional
        Positive initialization/regularization constant for forward and backward
        energies. Default is 0.1.
    w_init : ComplexArrayLike, optional
        Optional initial ladder coefficients of length ``M+1``. If None, initializes
        with zeros.
    safe_eps : float, optional
        Small positive floor used to avoid division by (near) zero and to keep the
        internal likelihood variables bounded. Default is 1e-12.

    Notes
    -----
    Signals and dimensions
    ~~~~~~~~~~~~~~~~~~~~~~
    This class operates on complex-valued sequences. For lattice order ``M``:

    - ``kappa_f`` and ``kappa_b`` have shape ``(M,)`` (reflection coefficients,
      updated directly through error feedback)
    - ``xi_f``, ``xi_b`` and ``gamma`` have shape ``(M+1,)``
    - ``error_b_prev`` has shape ``(M+1,)`` and stores the previous a posteriori
      backward-error vector.

    Output computation
    ~~~~~~~~~~~~~~~~~~
    The ladder starts from ``e = d(k)`` and, for each order ``m``, updates
    ``v_m`` by error feedback and removes ``v_m^*(k) b_m(k)`` from ``e``.
    The final value is the a posteriori error, returned in ``errors[k]``;
    ``outputs[k] = d[k] - errors[k]``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 7.5.
    """

    supports_complex: bool = True

    lam: float
    epsilon: float
    n_sections: int
    safe_eps: float

    kappa_f: ArrayLike
    kappa_b: ArrayLike
    xi_f: ArrayLike
    xi_b: ArrayLike
    error_b_prev: ArrayLike

    v: ArrayLike

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[ComplexArrayLike] = None,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Lattice order M (number of sections). The ladder has M+1 coefficients.
        lambda_factor:
            Forgetting factor λ.
        epsilon:
            Regularization/initialization constant for energies.
        w_init:
            Optional initial ladder coefficients (length M+1). If None, zeros.
        safe_eps:
            Small positive floor used to avoid division by (near) zero.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)
        self.safe_eps = float(safe_eps)

        self.kappa_f = np.zeros(self.n_sections, dtype=complex)
        self.kappa_b = np.zeros(self.n_sections, dtype=complex)

        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon

        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)
        self.gamma = np.ones(self.n_sections + 1, dtype=float)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).ravel()
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.w = self.v.copy()
        self.w_history = []
        self._record_history()

    @validate_input
    def optimize(
        self,
        input_signal: ArrayLike,
        desired_signal: ArrayLike,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes LRLS-EF adaptation for paired sequences ``x[k]`` and ``d[k]``.

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
                Estimated output sequence ``y[k] = d[k] - e[k]``.
            - errors : ndarray of complex, shape ``(N,)``
                A posteriori error sequence ``e[k]``.
            - coefficients : ndarray
                Ladder coefficient history (mirrors ``self.v`` via ``self.w``).
            - error_type : str
                Set to ``"a_posteriori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` (see below).

        Extra (when return_internal_states=True)
        --------------------------------------
        xi_f : ndarray of float, shape ``(M+1,)``
            Final forward prediction-error energies.
        xi_b : ndarray of float, shape ``(M+1,)``
            Final backward prediction-error energies.
        kappa_f, kappa_b : ndarray of complex, shape ``(M,)``
            Final forward/backward reflection coefficients.
        """
        tic: float = time()

        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        eps = self.safe_eps

        for k in range(n_samples):
            # Values from time k-1 needed by the error-feedback recursions.
            xi_b_prev = self.xi_b.copy()
            gamma_prev = self.gamma.copy()

            err_f = complex(x_in[k])

            curr_b = np.zeros(self.n_sections + 1, dtype=complex)
            curr_b[0] = x_in[k]

            gamma = np.ones(self.n_sections + 1, dtype=float)

            for m in range(self.n_sections + 1):
                g_prev = max(gamma_prev[m], eps)
                self.xi_f[m] = self.lam * self.xi_f[m] + float(np.real(err_f * np.conj(err_f))) / g_prev
                energy_b = float(np.real(curr_b[m] * np.conj(curr_b[m])))
                self.xi_b[m] = self.lam * self.xi_b[m] + energy_b / max(gamma[m], eps)

                if m == self.n_sections:
                    break

                gamma[m + 1] = max(gamma[m] - energy_b / (self.xi_b[m] + eps), eps)

                b_prev = self.error_b_prev[m]

                # Error-feedback updates of the reflection coefficients.
                self.kappa_b[m] = self.kappa_b[m] + (
                    np.conj(err_f) * (b_prev - self.kappa_b[m] * err_f)
                ) / (g_prev * self.xi_f[m] + eps)
                self.kappa_f[m] = self.kappa_f[m] + (
                    np.conj(b_prev) * (err_f - self.kappa_f[m] * b_prev)
                ) / (g_prev * xi_b_prev[m] + eps)

                curr_b[m + 1] = b_prev - self.kappa_b[m] * err_f
                err_f = err_f - self.kappa_f[m] * b_prev

            # Ladder stage (error feedback), producing the a posteriori error.
            e_k = complex(d_in[k])
            for m in range(self.n_sections + 1):
                e_prior = e_k - np.conj(self.v[m]) * curr_b[m]
                self.v[m] = self.v[m] + curr_b[m] * np.conj(e_prior) / (
                    max(gamma[m], eps) * self.xi_b[m] + eps
                )
                e_k = e_k - np.conj(self.v[m]) * curr_b[m]

            outputs[k] = d_in[k] - e_k
            errors[k] = e_k

            self.gamma = gamma
            self.error_b_prev = curr_b

            self.w = self.v.copy()
            self._record_history()

        runtime_s = float(time() - tic)
        if verbose:
            print(f"[LRLSErrorFeedback] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "xi_f": self.xi_f.copy(),
                "xi_b": self.xi_b.copy(),
                "kappa_f": self.kappa_f.copy(),
                "kappa_b": self.kappa_b.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_posteriori",
            extra=extra,
        )
# EOF