# blind.sato.py
#
#       Implements the Sato algorithm for COMPLEX valued data.
#       (Algorithm 13.3 - book: Adaptive Filtering: Algorithms and Practical
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


class Sato(AdaptiveFilter):
    """
    Implements the Sato algorithm for blind adaptive filtering with complex-valued data.

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
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        step_size:
            Adaptation step size.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
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
        Executes the Sato blind adaptive algorithm.

        Parameters
        ----------
        input_signal:
            Input signal to be filtered.
        desired_signal:
            Ignored (kept only for API standardization).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns internal signals in result.extra.
        safe_eps:
            Small epsilon used to avoid division by zero.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                Sato error defined here as: e[k] = y[k] - gamma * sign(y[k]).
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "blind_sato".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["sato_sign_track"]:
            Track of sign(y[k]) = y[k]/|y[k]| (with safe handling around zero), length N.
        extra["dispersion_constant"]:
            Scalar gamma used by the Sato criterion.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        num: float = float(np.mean(np.abs(x) ** 2))
        den: float = float(np.mean(np.abs(x)))
        dispersion_constant: float = float(num / (den + safe_eps)) if den > safe_eps else 0.0

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        sign_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            mag: float = float(np.abs(y_k))
            sato_sign: complex = (y_k / mag) if mag > safe_eps else (0.0 + 0.0j)

            if return_internal_states and sign_track is not None:
                sign_track[k] = sato_sign

            e_k: complex = y_k - sato_sign * dispersion_constant
            errors[k] = e_k

            self.w = self.w - self.step_size * np.conj(e_k) * x_k
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[Sato] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "sato_sign_track": sign_track,
                "dispersion_constant": dispersion_constant,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_sato",
            extra=extra,
        )
# EOF