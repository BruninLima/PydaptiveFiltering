#  set_membership.affine_projection.py
#
#       Implements the Set-membership Affine-Projection (SM-AP) algorithm
#       for COMPLEX valued data.
#       (Algorithm 6.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SMAffineProjection(AdaptiveFilter):
    """
    Set-Membership Affine-Projection (SM-AP) adaptive filter (complex-valued).

    Supervised affine-projection algorithm with *set-membership* updating,
    following Diniz (Alg. 6.2). Coefficients are updated **only** when the
    magnitude of the most-recent a priori error exceeds a prescribed bound
    ``gamma_bar``. When an update occurs, the algorithm enforces a target
    a posteriori error vector (provided by ``gamma_bar_vector``).

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    gamma_bar : float
        Set-membership bound for the (most recent) a priori error magnitude.
        An update is performed only if ``|e[k]| > gamma_bar``.
    gamma_bar_vector : array_like of complex
        Target a posteriori error vector with shape ``(L + 1,)`` (stored
        internally as a column vector). This is algorithm-dependent and
        corresponds to the desired post-update constraint in Alg. 6.2.
    gamma : float
        Regularization factor ``gamma`` used in the affine-projection normal
        equations to improve numerical stability.
    L : int
        Data reuse factor (projection order). The affine-projection block size is
        ``P = L + 1``.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    At iteration ``k``, form the regressor block matrix:

    - ``X(k) ∈ C^{(M+1) x (L+1)}``, whose columns are the most recent regressor
      vectors (newest in column 0).

    The affine-projection output vector is:

    .. math::
        y_{ap}(k) = X^H(k) w(k) \\in \\mathbb{C}^{L+1}.

    Let the stacked desired vector be:

    .. math::
        d_{ap}(k) \\in \\mathbb{C}^{L+1},

    with newest sample at index 0. The a priori error vector is:

    .. math::
        e_{ap}(k) = d_{ap}(k) - y_{ap}(k).

    This implementation uses the *most recent* scalar component as the reported
    output and error:

    .. math::
        y[k] = y_{ap}(k)[0], \\qquad e[k] = e_{ap}(k)[0].

    Set-membership update rule
        Update **only if**:

        .. math::
            |e[k]| > \\bar{\\gamma}.

        When updating, solve the regularized system:

        .. math::
            (X^H(k)X(k) + \\gamma I_{L+1})\\, s(k) =
            \\bigl(e_{ap}(k) - \\bar{\\gamma}_{vec}^*(k)\\bigr),

        and update the coefficients as:

        .. math::
            w(k+1) = w(k) + X(k)\\, s(k).

        Here ``\\bar{\\gamma}_{vec}`` is provided by ``gamma_bar_vector`` (stored
        as a column vector); complex conjugation is applied to match the internal
        conjugate-domain formulation used in the implementation.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 6.2.
    """
    supports_complex: bool = True

    gamma_bar: float
    gamma_bar_vector: np.ndarray
    gamma: float
    L: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        gamma_bar: float,
        gamma_bar_vector: Union[np.ndarray, list],
        gamma: float,
        L: int,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)

        self.n_coeffs = int(self.filter_order + 1)

        gvec = np.asarray(gamma_bar_vector, dtype=complex).ravel()
        if gvec.size != (self.L + 1):
            raise ValueError(
                f"gamma_bar_vector must have size L+1 = {self.L + 1}, got {gvec.size}"
            )
        self.gamma_bar_vector = gvec.reshape(-1, 1)

        self.regressor_matrix = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)

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
        Executes the SM-AP adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints total runtime and update count after completion.
        return_internal_states : bool, optional
            If True, includes the full a priori AP error-vector trajectory in
            ``result.extra`` as ``"errors_vector"`` with shape ``(N, L + 1)``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = y_{ap}(k)[0]``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = e_{ap}(k)[0]``.
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
                - ``"errors_vector"`` : ndarray of complex, shape ``(N, L + 1)``
                    Full affine-projection a priori error vectors over time.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)
        Lp1: int = int(self.L + 1)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)
        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        errors_vec_track: Optional[np.ndarray] = (
            np.zeros((n_samples, Lp1), dtype=complex) if return_internal_states else None
        )

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])
        prefixed_desired: np.ndarray = np.concatenate([np.zeros(self.L, dtype=complex), d])

        for k in range(n_samples):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]

            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            output_ap_conj = (self.regressor_matrix.conj().T) @ w_current

            desired_slice = prefixed_desired[k + self.L : stop : -1]
            error_ap_conj = desired_slice.conj().reshape(-1, 1) - output_ap_conj

            yk = output_ap_conj[0, 0]
            ek = error_ap_conj[0, 0]

            outputs[k] = yk
            errors[k] = ek
            if return_internal_states and errors_vec_track is not None:
                errors_vec_track[k, :] = error_ap_conj.ravel()

            if np.abs(ek) > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True

                R = (self.regressor_matrix.conj().T @ self.regressor_matrix) + self.gamma * np.eye(Lp1)
                b = error_ap_conj - self.gamma_bar_vector.conj()

                try:
                    step = np.linalg.solve(R, b)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(R) @ b

                w_current = w_current + (self.regressor_matrix @ step)

            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[SM-AP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.02f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra["errors_vector"] = errors_vec_track

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF
