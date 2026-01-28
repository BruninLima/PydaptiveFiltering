# blind.constant_modulus.py
#
#       Implements the Constant-Modulus algorithm for COMPLEX valued data.
#       (Algorithm 13.2 - book: Adaptive Filtering: Algorithms and Practical
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

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class CMA(AdaptiveFilter):
    """
    Implements the Constant-Modulus Algorithm (CMA) for blind adaptive filtering.

    Notes
    -----
    - This is a BLIND algorithm: it does not require desired_signal.
    - We keep `desired_signal=None` in `optimize` only for API standardization.
    """
    supports_complex: bool = True

    step_size: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.01,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Optional[Union[np.ndarray, list]] = None,
        verbose: bool = False,
        return_internal_states: bool = False,
        safe_eps: float = 1e-12,
    ) -> OptimizationResult:
        """
        Executes the Constant-Modulus Algorithm (CMA) weight update process.

        Parameters
        ----------
        input_signal:
            Input signal to be filtered.
        desired_signal:
            Ignored (kept only for API standardization).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, includes internal signals in result.extra.
        safe_eps:
            Small epsilon to avoid division by zero when estimating the dispersion constant.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                CMA error (|y[k]|^2 - R2).
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "blind_constant_modulus".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["dispersion_constant"]:
            R2 used by CMA.
        extra["instantaneous_phi"]:
            Trajectory of phi[k] = 2*e[k]*conj(y[k]) (complex), length N.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=float)
        denom: float = float(np.mean(np.abs(x) ** 2))
        if denom < safe_eps:
            desired_level: float = 0.0
        else:
            desired_level = float(np.mean(np.abs(x) ** 4) / (denom + safe_eps))

        phi_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            e_k: float = float((np.abs(y_k) ** 2) - desired_level)
            errors[k] = e_k

            phi_k: complex = complex(2.0 * e_k * np.conj(y_k))
            if return_internal_states and phi_track is not None:
                phi_track[k] = phi_k

            self.w = self.w - self.step_size * phi_k * x_k
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[CMA] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "dispersion_constant": desired_level,
                "instantaneous_phi": phi_track,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_constant_modulus",
            extra=extra,
        )
# EOF