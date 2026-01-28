# blind.godard.py
#
#       Implements the Godard algorithm for COMPLEX valued data.
#       (Algorithm 13.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace.wam@gmail.com
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class Godard(AdaptiveFilter):
    """
    Implements the Godard algorithm for blind adaptive filtering with complex-valued data.

    This is a blind adaptation criterion that does not require a desired signal.
    A `desired_signal=None` parameter is accepted only to keep a unified API signature
    across the library.
    """
    supports_complex: bool = True

    step_size: float
    p: int
    q: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.01,
        p_exponent: int = 2,
        q_exponent: int = 2,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        step_size:
            Adaptation step size.
        p_exponent:
            Exponent p used by the Godard cost (typically p=2).
        q_exponent:
            Exponent q used by the Godard cost (typically q=2).
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.p = int(p_exponent)
        self.q = int(q_exponent)
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
        Executes the Godard adaptive algorithm.

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
            Small epsilon used to avoid divisions by zero and unstable powers.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k].
            errors:
                Godard error defined here as: e[k] = |y[k]|^q - Rq.
            coefficients:
                History of coefficients stored in the base class.
            error_type:
                "blind_godard".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["phi_gradient"]:
            Trajectory of the instantaneous gradient term used for weight update, length N.
        extra["dispersion_constant"]:
            Scalar Rq used by the criterion.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        num: float = float(np.mean(np.abs(x) ** (2 * self.q)))
        den: float = float(np.mean(np.abs(x) ** self.q))
        desired_level: float = float(num / (den + safe_eps)) if den > safe_eps else 0.0

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=float)

        phi_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=complex) if return_internal_states else None

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.filter_order + 1][::-1]

            y_k: complex = complex(np.dot(np.conj(self.w), x_k))
            outputs[k] = y_k

            e_k: float = float((np.abs(y_k) ** self.q) - desired_level)
            errors[k] = e_k

            if np.abs(y_k) > safe_eps:
                phi_k: complex = complex(
                    self.p
                    * self.q
                    * (e_k ** (self.p - 1))
                    * (np.abs(y_k) ** (self.q - 2))
                    * np.conj(y_k)
                )
            else:
                phi_k = 0.0 + 0.0j

            if return_internal_states and phi_track is not None:
                phi_track[k] = phi_k

            self.w = self.w - (self.step_size * phi_k * x_k) / 2.0
            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[Godard] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "phi_gradient": phi_track,
                "dispersion_constant": desired_level,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_godard",
            extra=extra,
        )
# EOF