#  lms.affine_projection.py
#
#       Implements the Complex Affine-Projection algorithm for COMPLEX valued data.
#       (Algorithm 4.6 - book: Adaptive Filtering: Algorithms and Practical
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
from time import perf_counter
from typing import Optional
from pydaptivefiltering.base import AdaptiveFilter, validate_input, OptimizationResult
from pydaptivefiltering._utils.typing import ArrayLike



class AffineProjection(AdaptiveFilter):
    """
    Complex Affine-Projection Algorithm (APA) adaptive filter.

    Affine-projection LMS-type algorithm that reuses the last ``L+1`` regressor
    vectors to accelerate convergence relative to LMS/NLMS, following Diniz
    (Alg. 4.6). Per iteration, the method solves a small linear system of size
    ``(L+1) x (L+1)``.

    Parameters
    ----------
    filter_order : int
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
    step_size : float, optional
        Adaptation step size (relaxation factor) ``mu``. Default is 1e-2.
    gamma : float, optional
        Diagonal loading (regularization) ``gamma`` applied to the projection
        correlation matrix for numerical stability. Default is 1e-6.
    L : int, optional
        Reuse factor (projection order). The algorithm uses ``L + 1`` most recent
        regressors. Default is 2.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    At iteration ``k``, form the projection matrix and desired vector:

    - ``X(k) ∈ C^{(L+1) x (M+1)}``, whose rows are regressor vectors, with the most
      recent regressor at row 0.
    - ``d_vec(k) ∈ C^{L+1}``, stacking the most recent desired samples, with
      ``d[k]`` at index 0.

    The projection output and error vectors are:

    .. math::
        y_{vec}(k) = X(k)\\,w^*(k) \\in \\mathbb{C}^{L+1},

    .. math::
        e_{vec}(k) = d_{vec}(k) - y_{vec}(k).

    The update direction ``u(k)`` is obtained by solving the regularized system:

    .. math::
        (X(k)X^H(k) + \\gamma I_{L+1})\\,u(k) = e_{vec}(k),

    and the coefficient update is:

    .. math::
        w(k+1) = w(k) + \\mu X^H(k)\\,u(k).

    This implementation returns only the *most recent* scalar components:

    - ``y[k] = y_vec(k)[0]``
    - ``e[k] = e_vec(k)[0]``

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 4.6.
    """

    supports_complex: bool = True

    step_size: float
    gamma: float
    memory_length: int

    def __init__(
        self,
        filter_order: int,
        step_size: float = 1e-2,
        gamma: float = 1e-6,
        L: int = 2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.step_size = float(step_size)
        self.gamma = float(gamma)
        self.memory_length = int(L)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Affine Projection adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"last_regressor_matrix"`` (``X(k)``) and
            ``"last_correlation_matrix"`` (``X(k)X^H(k) + gamma I``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar output sequence, ``y[k] = y_vec(k)[0]``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = e_vec(k)[0]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        tic: float = perf_counter()

        dtype = complex
        x = np.asarray(input_signal, dtype=dtype).ravel()
        d = np.asarray(desired_signal, dtype=dtype).ravel()
        
        n_samples: int = int(x.size)
        m: int = int(self.filter_order)
        L: int = int(self.memory_length)

        outputs: np.ndarray = np.zeros(n_samples, dtype=dtype)
        errors: np.ndarray = np.zeros(n_samples, dtype=dtype)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=dtype)
        x_padded[m:] = x

        X_matrix: np.ndarray = np.zeros((L + 1, m + 1), dtype=dtype)
        D_vector: np.ndarray = np.zeros(L + 1, dtype=dtype)

        last_corr: Optional[np.ndarray] = None

        eye_L: np.ndarray = np.eye(L + 1, dtype=dtype)

        for k in range(n_samples):
            X_matrix[1:] = X_matrix[:-1]
            X_matrix[0] = x_padded[k : k + m + 1][::-1]

            D_vector[1:] = D_vector[:-1]
            D_vector[0] = d[k]

            Y_vector: np.ndarray = X_matrix @ self.w.conj()
            E_vector: np.ndarray = D_vector - Y_vector

            outputs[k] = Y_vector[0]
            errors[k] = E_vector[0]

            corr_matrix: np.ndarray = (X_matrix @ X_matrix.conj().T) + (self.gamma * eye_L)
            last_corr = corr_matrix

            try:
                u: np.ndarray = np.linalg.solve(corr_matrix, E_vector)
            except np.linalg.LinAlgError:
                u = np.linalg.pinv(corr_matrix) @ E_vector

            self.w = self.w + self.step_size * (X_matrix.T @ u.conj())
            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[AffineProjection] Completed in {runtime_s * 1000:.02f} ms")

        extra = None
        if return_internal_states:
            extra = {
                "last_regressor_matrix": X_matrix.copy(),
                "last_correlation_matrix": None if last_corr is None else last_corr.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF