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

from time import perf_counter
from typing import Optional, Union

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, validate_input, OptimizationResult


ArrayLike = Union[np.ndarray, list]


class AffineProjection(AdaptiveFilter):
    """
    Complex Affine Projection Algorithm (APA).

    Implements Algorithm 4.6 (Diniz) using an affine-projection update with data reuse.

    Notes
    -----
    - This implementation supports complex-valued data (supports_complex=True).
    - The base decorator `@validate_input` allows calling optimize with:
        * optimize(input_signal=..., desired_signal=...)
        * optimize(x=..., d=...)
        * optimize(x, d)
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
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        step_size:
            Step-size / relaxation factor (mu).
        gamma:
            Diagonal loading regularization to ensure invertibility.
        L:
            Data reuse factor (projection order). Uses L+1 past regressors.
        w_init:
            Optional initial weights (length M+1). If None, initializes to zeros.
        """
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
        Run APA adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns the last regressor matrix and last correlation matrix in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k] (a priori).
            errors:
                Error e[k] = d[k] - y[k] (a priori).
            coefficients:
                Coefficient history (self.w_history) as a 2D array.
            error_type:
                "a_priori".
            extra (optional):
                last_regressor_matrix, last_correlation_matrix.
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal).ravel()
        d: np.ndarray = np.asarray(desired_signal).ravel()

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

            self.w = self.w + self.step_size * (X_matrix.conj().T @ u)
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