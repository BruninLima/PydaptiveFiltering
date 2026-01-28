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
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


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
    w_init : array_like of complex, optional
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

    - ``delta`` and ``delta_v`` have shape ``(M+1,)``
    - ``xi_f`` and ``xi_b`` have shape ``(M+2,)`` (energies per section plus guard)
    - ``error_b_prev`` has shape ``(M+2,)`` and stores the previous backward-error
      vector used for the error-feedback recursion.
    - At each time k, the ladder regressor is the backward-error vector
      ``curr_b[:M+1]``.

    Output computation
    ~~~~~~~~~~~~~~~~~~
    The estimated output is formed as a ladder combination:

    .. math::
        y(k) = \\mathbf{v}^H(k)\\, \\mathbf{b}(k),

    where :math:`\\mathbf{b}(k)` corresponds to ``curr_b[:M+1]`` and
    :math:`\\mathbf{v}(k)` is the ladder coefficient vector ``self.v``.
    The reported error is the output error :math:`e(k)=d(k)-y(k)`.

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

    delta: np.ndarray
    xi_f: np.ndarray
    xi_b: np.ndarray
    error_b_prev: np.ndarray

    v: np.ndarray
    delta_v: np.ndarray

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        safe_eps: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Lattice order M (number of sections). Ladder has M+1 coefficients.
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

        self.delta = np.zeros(self.n_sections + 1, dtype=complex)

        self.xi_f = np.ones(self.n_sections + 2, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 2, dtype=float) * self.epsilon

        self.error_b_prev = np.zeros(self.n_sections + 2, dtype=complex)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).ravel()
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

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
                Estimated output sequence ``y[k]``.
            - errors : ndarray of complex, shape ``(N,)``
                Output error sequence ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray
                Ladder coefficient history (mirrors ``self.v`` via ``self.w``).
            - error_type : str
                Set to ``"output_error"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` (see below).

        Extra (when return_internal_states=True)
        --------------------------------------
        xi_f : ndarray of float, shape ``(M+2,)``
            Final forward prediction-error energies.
        xi_b : ndarray of float, shape ``(M+2,)``
            Final backward prediction-error energies.
        delta : ndarray of complex, shape ``(M+1,)``
            Final lattice delta (reflection-like) state.
        delta_v : ndarray of complex, shape ``(M+1,)``
            Final ladder delta state used to compute ``v``.
        """
        tic: float = time()

        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        eps = self.safe_eps

        for k in range(n_samples):
            err_f = complex(x_in[k])

            curr_b = np.zeros(self.n_sections + 2, dtype=complex)
            curr_b[0] = x_in[k]

            energy_x = float(np.real(x_in[k] * np.conj(x_in[k])))
            self.xi_f[0] = self.lam * self.xi_f[0] + energy_x
            self.xi_b[0] = self.xi_f[0]

            g = 1.0

            for m in range(self.n_sections + 1):
                denom_g = max(g, eps)

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(err_f)) / denom_g
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + eps)
                kappa_b = self.delta[m] / (self.xi_f[m] + eps)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_b[m + 1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m + 1] = (
                    self.lam * self.xi_f[m + 1]
                    + float(np.real(new_err_f * np.conj(new_err_f))) / denom_g
                )
                self.xi_b[m + 1] = (
                    self.lam * self.xi_b[m + 1]
                    + float(np.real(curr_b[m + 1] * np.conj(curr_b[m + 1]))) / denom_g
                )

                energy_b_curr = float(np.real(curr_b[m] * np.conj(curr_b[m])))
                g = g - (energy_b_curr / (self.xi_b[m] + eps))
                g = max(g, eps)

                err_f = new_err_f

            y_k = complex(np.vdot(self.v, curr_b[: self.n_sections + 1]))
            outputs[k] = y_k
            e_k = complex(d_in[k] - y_k)
            errors[k] = e_k

            g_ladder = 1.0
            for m in range(self.n_sections + 1):
                denom_gl = max(g_ladder, eps)

                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (curr_b[m] * np.conj(d_in[k])) / denom_gl
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + eps)

                energy_b = float(np.real(curr_b[m] * np.conj(curr_b[m])))
                g_ladder = g_ladder - (energy_b / (self.xi_b[m] + eps))
                g_ladder = max(g_ladder, eps)

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
                "delta": self.delta.copy(),
                "delta_v": self.delta_v.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF