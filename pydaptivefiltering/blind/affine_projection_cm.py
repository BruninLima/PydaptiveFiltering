# blind.affine_projection_cm.py
#
#       Implements the Complex Affine-Projection Constant-Modulus algorithm
#       for COMPLEX valued data.
#       (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermeodeoliveirapinto@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class AffineProjectionCM(AdaptiveFilter):
    """
    Implements the Affine-Projection Constant-Modulus (AP-CM) algorithm
    for blind adaptive filtering.

    Notes
    -----
    - This is a BLIND algorithm: it does not require desired_signal.
    - We still accept `desired_signal=None` in `optimize` to keep a unified API.
    """
    supports_complex: bool = True
    step_size: float
    memory_length: int
    gamma: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.1,
        memory_length: int = 2,
        gamma: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.memory_length = int(memory_length)
        self.gamma = float(gamma)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Optional[Union[np.ndarray, list]] = None,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the Affine-Projection Constant-Modulus (AP-CM) algorithm.

        Parameters
        ----------
        input_signal:
            The input signal to be filtered.
        desired_signal:
            Ignored (kept only for API standardization).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns selected internal values in `extra`.

        Returns
        -------
        OptimizationResult
            outputs:
                y[k] = first component of the projection output vector.
            errors:
                e[k] = first component of the CM error vector.
            coefficients:
                coefficient history stored in the base class.
            error_type:
                set to "blind_constant_modulus".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["last_update_factor"]:
            Solution of the (regularized) linear system at the last iteration.
        extra["last_regressor_matrix"]:
            Final regressor matrix (shape n_coeffs x (memory_length+1)).
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        L: int = int(self.memory_length)

        regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, L + 1), dtype=complex)
        I_reg: np.ndarray = (self.gamma * np.eye(L + 1)).astype(complex)

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        last_update_factor: Optional[np.ndarray] = None

        for k in range(n_samples):
            regressor_matrix[:, 1:] = regressor_matrix[:, :-1]
            regressor_matrix[:, 0] = x_padded[k : k + self.filter_order + 1][::-1]

            output_ap: np.ndarray = np.dot(np.conj(regressor_matrix).T, self.w)

            abs_out: np.ndarray = np.abs(output_ap)
            desired_level: np.ndarray = np.zeros_like(output_ap, dtype=complex)
            np.divide(output_ap, abs_out, out=desired_level, where=abs_out > 1e-12)

            error_ap: np.ndarray = desired_level - output_ap

            phi: np.ndarray = np.dot(np.conj(regressor_matrix).T, regressor_matrix) + I_reg
            update_factor: np.ndarray = np.linalg.solve(phi, error_ap)
            last_update_factor = update_factor

            self.w = self.w + self.step_size * np.dot(regressor_matrix, update_factor)

            outputs[k] = output_ap[0]
            errors[k] = error_ap[0]

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[AffineProjectionCM] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "last_update_factor": last_update_factor,
                "last_regressor_matrix": regressor_matrix.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_constant_modulus",
            extra=extra,
        )
# EOF