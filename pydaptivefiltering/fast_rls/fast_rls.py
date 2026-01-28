# fast_rls.fast_rls.py
#
#       Implements the Fast Transversal RLS algorithm for COMPLEX valued data.
#       (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical
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


class FastRLS(AdaptiveFilter):
    """
    Fast Transversal Recursive Least-Squares (FT-RLS) algorithm (complex-valued).

    The Fast Transversal RLS (also called Fast RLS) is a computationally
    efficient alternative to standard RLS. By exploiting shift-structure in the
    regressor and using coupled forward/backward linear prediction recursions,
    it reduces the per-sample complexity from :math:`O(M^2)` (standard RLS) to
    approximately :math:`O(M)`.

    This implementation follows Diniz (Alg. 8.1) and maintains internal state
    for forward and backward predictors, as well as the conversion (likelihood)
    variable :math:`\\gamma(k)` that maps a priori to a posteriori quantities.

    Parameters
    ----------
    filter_order : int
        FIR filter order ``M``. The number of coefficients is ``M + 1``.
    forgetting_factor : float, optional
        Exponential forgetting factor ``lambda``. Typical values are in
        ``[0.95, 1.0]``; values closer to 1 give longer memory. Default is 0.99.
    epsilon : float, optional
        Positive initialization for the minimum prediction-error energies
        (regularization), used as :math:`\\xi_{\\min}(0)` in the recursions.
        Default is 0.1.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.
    safe_eps : float, optional (keyword-only)
        Small constant used to guard divisions in internal recursions when
        denominators approach zero. Default is 1e-30.

    Notes
    -----
    Convention
    ~~~~~~~~~~
    At time ``k``, the regressor is formed (most recent sample first) as:

    .. math::
        x_k = [x[k], x[k-1], \\ldots, x[k-M]]^T.

    A priori vs a posteriori
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The a priori output and error are:

    .. math::
        y(k) = w^H(k-1) x_k, \\qquad e(k) = d(k) - y(k).

    This implementation also computes the a posteriori error using the
    conversion variable :math:`\\gamma(k)` (from the FT-RLS recursions):

    .. math::
        e_{\\text{post}}(k) = \\gamma(k)\\, e(k), \\qquad
        y_{\\text{post}}(k) = d(k) - e_{\\text{post}}(k).

    The main-filter coefficient update uses the normalized gain-like vector
    produced by the transversal recursions (``phi_hat_n`` in the code):

    .. math::
        w(k) = w(k-1) + \\phi(k)\\, e_{\\text{post}}^*(k),

    where :math:`\\phi(k)` corresponds to the internal vector ``phi_hat_n``.

    Returned internals
    ~~~~~~~~~~~~~~~~~~
    The method always returns a posteriori sequences in ``extra``:
    ``outputs_posteriori`` and ``errors_posteriori``. If
    ``return_internal_states=True``, it additionally returns tracks of
    ``gamma`` and the forward minimum prediction-error energy ``xi_min_f``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 8.1.
    """
    supports_complex: bool = True
    forgetting_factor: float
    epsilon: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        forgetting_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        *,
        safe_eps: float = 1e-30,
    ) -> None:
        super().__init__(filter_order=filter_order, w_init=w_init)
        self.forgetting_factor = float(forgetting_factor)
        self.epsilon = float(epsilon)
        self.n_coeffs = int(filter_order + 1)
        self._safe_eps = float(safe_eps)

        self.w = np.asarray(self.w, dtype=np.complex128)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the FT-RLS adaptation loop.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired/reference sequence ``d[k]`` with shape ``(N,)`` (will be
            flattened). Must have the same length as ``input_signal``.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes additional internal trajectories in
            ``result.extra``:
            - ``"gamma"``: ndarray of float, shape ``(N,)`` with :math:`\\gamma(k)`.
            - ``"xi_min_f"``: ndarray of float, shape ``(N,)`` with the forward
              minimum prediction-error energy :math:`\\xi_{f,\\min}(k)`.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                A priori output sequence ``y[k] = w^H(k-1) x_k``.
            - errors : ndarray of complex, shape ``(N,)``
                A priori error sequence ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict
                Always includes:
                - ``"outputs_posteriori"``: ndarray of complex, shape ``(N,)``.
                - ``"errors_posteriori"``: ndarray of complex, shape ``(N,)``.
                Additionally includes ``"gamma"`` and ``"xi_min_f"`` if
                ``return_internal_states=True``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=np.complex128).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=np.complex128).ravel()

        n_samples: int = int(x.size)
        m_plus_1: int = int(self.filter_order + 1)

        outputs: np.ndarray = np.zeros(n_samples, dtype=np.complex128)
        errors: np.ndarray = np.zeros(n_samples, dtype=np.complex128)
        outputs_post: np.ndarray = np.zeros(n_samples, dtype=np.complex128)
        errors_post: np.ndarray = np.zeros(n_samples, dtype=np.complex128)

        gamma_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        xi_f_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None

        w_f: np.ndarray = np.zeros(m_plus_1, dtype=np.complex128)
        w_b: np.ndarray = np.zeros(m_plus_1, dtype=np.complex128)
        phi_hat_n: np.ndarray = np.zeros(m_plus_1, dtype=np.complex128)

        gamma_n: float = 1.0
        xi_min_f_prev: float = float(self.epsilon)
        xi_min_b: float = float(self.epsilon)

        x_padded: np.ndarray = np.zeros(n_samples + m_plus_1, dtype=np.complex128)
        x_padded[m_plus_1:] = x

        lam = float(self.forgetting_factor)
        eps = float(self._safe_eps)

        for k in range(n_samples):
            regressor: np.ndarray = x_padded[k : k + m_plus_1 + 1][::-1]

            e_f_priori: np.complex128 = regressor[0] - np.dot(w_f.conj(), regressor[1:])
            e_f_post: np.complex128 = e_f_priori * gamma_n

            xi_min_f_curr: float = float(lam * xi_min_f_prev + np.real(e_f_priori * np.conj(e_f_post)))

            den_phi = lam * xi_min_f_prev
            if abs(den_phi) < eps:
                den_phi = np.copysign(eps, den_phi if den_phi != 0 else 1.0)
            phi_gain: np.complex128 = e_f_priori / den_phi

            phi_hat_n_plus_1: np.ndarray = np.zeros(m_plus_1 + 1, dtype=np.complex128)
            phi_hat_n_plus_1[1:] = phi_hat_n
            phi_hat_n_plus_1[0] += phi_gain
            phi_hat_n_plus_1[1:] -= phi_gain * w_f

            w_f = w_f + phi_hat_n * np.conj(e_f_post)

            den_g = xi_min_f_curr
            if abs(den_g) < eps:
                den_g = np.copysign(eps, den_g if den_g != 0 else 1.0)
            gamma_n_plus_1: float = float((lam * xi_min_f_prev * gamma_n) / den_g)

            e_b_priori: np.complex128 = lam * xi_min_b * phi_hat_n_plus_1[-1]

            den_gamma = np.real((1.0 / gamma_n_plus_1) - (phi_hat_n_plus_1[-1] * np.conj(e_b_priori)))
            if abs(den_gamma) < eps:
                den_gamma = np.copysign(eps, den_gamma if den_gamma != 0 else 1.0)
            gamma_n = float(1.0 / den_gamma)

            e_b_post: np.complex128 = e_b_priori * gamma_n
            xi_min_b = float(lam * xi_min_b + np.real(e_b_post * np.conj(e_b_priori)))

            phi_hat_n = phi_hat_n_plus_1[:-1] + phi_hat_n_plus_1[-1] * w_b
            w_b = w_b + phi_hat_n * np.conj(e_b_post)

            y_k: np.complex128 = np.dot(self.w.conj(), regressor[:-1])
            outputs[k] = y_k

            e_k: np.complex128 = d[k] - y_k
            errors[k] = e_k

            errors_post[k] = e_k * gamma_n
            outputs_post[k] = d[k] - errors_post[k]

            self.w = self.w + phi_hat_n * np.conj(errors_post[k])
            self._record_history()

            if return_internal_states and gamma_track is not None and xi_f_track is not None:
                gamma_track[k] = gamma_n
                xi_f_track[k] = xi_min_f_curr

            xi_min_f_prev = xi_min_f_curr

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[FastRLS] Completed in {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "outputs_posteriori": outputs_post,
            "errors_posteriori": errors_post,
        }
        if return_internal_states:
            extra.update({"gamma": gamma_track, "xi_min_f": xi_f_track})

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF