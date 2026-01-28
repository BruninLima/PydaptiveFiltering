#  lms.nlms.py
#
#       Implements the Normalized LMS algorithm for COMPLEX valued data.
#       (Algorithm 4.3 - book: Adaptive Filtering: Algorithms and Practical
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

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class NLMS(AdaptiveFilter):
    """
    Complex NLMS (Normalized Least-Mean Squares).

    Implements the normalized LMS recursion (Algorithm 4.3 - Diniz) for adaptive
    FIR filtering. The update uses a step normalized by the regressor energy:

        y[k] = w[k]^H x_k
        e[k] = d[k] - y[k]
        mu_k = mu / (||x_k||^2 + gamma)
        w[k+1] = w[k] + mu_k * conj(e[k]) * x_k

    Notes
    -----
    - Complex-valued implementation (supports_complex = True).
    - Uses the unified base API via `@validate_input`.
    - `gamma` is a regularization constant to avoid division by zero.
    """

    supports_complex: bool = True

    step_size: float
    gamma: float

    def __init__(
        self,
        filter_order: int,
        step_size: float = 1e-2,
        gamma: float = 1e-6,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1).
        step_size:
            Base step-size (mu).
        gamma:
            Regularization to avoid division by (near) zero.
        w_init:
            Optional initial coefficients (length M+1). If None, zeros.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)
        self.step_size = float(step_size)
        self.gamma = float(gamma)

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run NLMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "a_priori".
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(x.size)
        m: int = int(self.filter_order)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        x_padded: np.ndarray = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + m + 1][::-1]

            y_k: complex = complex(np.vdot(self.w, x_k))
            outputs[k] = y_k

            e_k: complex = d[k] - y_k
            errors[k] = e_k

            norm_xk: float = float(np.vdot(x_k, x_k).real)
            mu_k: float = self.step_size / (norm_xk + self.gamma)

            self.w = self.w + mu_k * np.conj(e_k) * x_k

            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[NLMS] Completed in {runtime_s * 1000:.03f} ms")

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
        )
# EOF