#  lms.tdomain_dft.py
#
#       Implements the Transform-Domain LMS algorithm, based on the Discrete
#       Fourier Transform (DFT), for COMPLEX valued data.
#       (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union, Dict, Any, List

import numpy as np
from scipy.fft import fft, ifft

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class TDomainDFT(AdaptiveFilter):
    """
    Transform-Domain LMS using a DFT (complex-valued).

    Implements a transform-domain LMS variant (Algorithm 4.4 - Diniz) where the
    regressor is transformed via a unitary DFT:

        z_k = FFT(x_k) / sqrt(N)
        y[k] = w_z^H z_k
        e[k] = d[k] - y[k]
        P_z[k] = alpha * |z_k|^2 + (1-alpha) * P_z[k-1]
        w_z <- w_z + mu * conj(e[k]) * z_k / (gamma + P_z[k])

    Time-domain coefficients are recovered by:
        w = IFFT(w_z) * sqrt(N)

    Library conventions
    -------------------
    - Complex-valued implementation (`supports_complex=True`).
    - `OptimizationResult.coefficients` stores time-domain coefficient history (self.w_history).
    - Transform-domain coefficient history is provided in `result.extra["coefficients_dft"]`
      when requested.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        gamma: float,
        alpha: float,
        initial_power: float,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1). The DFT size is N = M+1.
        gamma:
            Small positive constant to avoid division by (near) zero in each bin.
        alpha:
            Smoothing factor for power estimation (typically close to 1).
        initial_power:
            Initial power estimate for all bins.
        step_size:
            Step-size (mu).
        w_init:
            Optional initial coefficients in time domain (length M+1). If None, zeros.
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.step_size = float(step_size)

        self.N = int(self.filter_order + 1)
        self._sqrtN = float(np.sqrt(self.N))

        self.w_dft = fft(np.asarray(self.w, dtype=complex)) / self._sqrtN

        self.power_vector = np.full(self.N, float(initial_power), dtype=float)

        self._w_history_dft: List[np.ndarray] = [self.w_dft.copy()]

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Transform-Domain LMS (DFT) adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns extra sequences such as DFT coefficients history and final power vector.

        Returns
        -------
        OptimizationResult
            outputs:
                Filter output y[k] (a priori).
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                Time-domain coefficient history stored in the base class.
            error_type:
                "a_priori".
        """
        t0 = perf_counter()

        x = np.asarray(input_signal, dtype=complex).ravel()
        d = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d.size)
        m = int(self.filter_order)

        outputs = np.zeros(n_samples, dtype=complex)
        errors = np.zeros(n_samples, dtype=complex)

        x_padded = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        w_hist_dft: List[np.ndarray] = [self.w_dft.copy()]

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]
            z_k = fft(x_k) / self._sqrtN

            self.power_vector = (
                self.alpha * np.real(z_k * np.conj(z_k)) + (1.0 - self.alpha) * self.power_vector
            )

            y_k = complex(np.vdot(self.w_dft, z_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            denom = self.gamma + self.power_vector
            self.w_dft = self.w_dft + self.step_size * np.conj(e_k) * (z_k / denom)

            self.w = ifft(self.w_dft) * self._sqrtN

            self._record_history()
            w_hist_dft.append(self.w_dft.copy())

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[TDomainDFT] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "coefficients_dft": np.asarray(w_hist_dft),
                "power_vector_last": self.power_vector.copy(),
                "sqrtN": self._sqrtN,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF