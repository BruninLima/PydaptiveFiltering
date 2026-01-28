#  lms.tdomain.py
#
#       Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
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

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input

ArrayLike = Union[np.ndarray, list]


class TDomainLMS(AdaptiveFilter):
    """
    Generic Transform-Domain LMS using a user-provided unitary transform matrix T.

    This is a transform-domain LMS variant (Algorithm 4.4 - Diniz). Given a transform
    z_k = T x_k and transform-domain weights w_T, the recursion is:

        y[k] = w_T^H z_k
        e[k] = d[k] - y[k]
        P_z[k] = alpha * |z_k|^2 + (1-alpha) * P_z[k-1]
        w_T <- w_T + mu * conj(e[k]) * z_k / (gamma + P_z[k])

    For library consistency, this implementation also exposes time-domain weights:

        w_time = T^H w_T

    Notes
    -----
    - Complex-valued implementation (`supports_complex=True`).
    - `OptimizationResult.coefficients` stores time-domain coefficient history (self.w_history).
    - Transform-domain coefficient history is returned in `result.extra["coefficients_transform"]`
      when requested.
    - `transform_matrix` is expected to be unitary (T^H T = I). If it is not, the mapping
      back to time domain is not the true inverse transform.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        gamma: float,
        alpha: float,
        initial_power: float,
        transform_matrix: np.ndarray,
        step_size: float = 1e-2,
        w_init: Optional[ArrayLike] = None,
        *,
        assume_unitary: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR order M (number of taps is M+1). Transform size must be (M+1, M+1).
        gamma:
            Small positive constant to avoid division by (near) zero in each bin.
        alpha:
            Smoothing factor for power estimation (typically close to 1).
        initial_power:
            Initial power estimate for all transform bins.
        transform_matrix:
            Transform matrix T of shape (M+1, M+1). Typically unitary.
        step_size:
            Step-size (mu).
        w_init:
            Optional initial coefficients in time domain (length M+1). If None, zeros.
        assume_unitary:
            If True, uses w_time = T^H w_T. If False, uses a least-squares mapping
            w_time = pinv(T)^H w_T (slower, but works for non-unitary transforms).
        """
        super().__init__(filter_order=int(filter_order), w_init=w_init)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.step_size = float(step_size)

        self.N = int(self.filter_order + 1)

        T = np.asarray(transform_matrix, dtype=complex)
        if T.shape != (self.N, self.N):
            raise ValueError(f"transform_matrix must have shape {(self.N, self.N)}. Got {T.shape}.")

        self.T = T
        self._assume_unitary = bool(assume_unitary)

        # transform-domain weights (start from time-domain w)
        self.w_T = self.T @ np.asarray(self.w, dtype=complex)

        # power estimate per transform bin
        self.power_vector = np.full(self.N, float(initial_power), dtype=float)

        # optional transform-domain history
        self._w_history_T: List[np.ndarray] = [self.w_T.copy()]

    def _to_time_domain(self, w_T: np.ndarray) -> np.ndarray:
        """Map transform-domain weights to time-domain weights."""
        if self._assume_unitary:
            return self.T.conj().T @ w_T
        # fallback for non-unitary transforms (more expensive)
        T_pinv = np.linalg.pinv(self.T)
        return T_pinv.conj().T @ w_T

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Run Transform-Domain LMS adaptation.

        Parameters
        ----------
        input_signal:
            Input signal x[k].
        desired_signal:
            Desired signal d[k].
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns transform-domain coefficient history and final power vector in result.extra.

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

        w_hist_T: List[np.ndarray] = [self.w_T.copy()]

        for k in range(n_samples):
            x_k = x_padded[k : k + m + 1][::-1]
            z_k = self.T @ x_k

            self.power_vector = (
                self.alpha * np.real(z_k * np.conj(z_k)) + (1.0 - self.alpha) * self.power_vector
            )

            y_k = complex(np.vdot(self.w_T, z_k))
            outputs[k] = y_k

            e_k = d[k] - y_k
            errors[k] = e_k

            denom = self.gamma + self.power_vector
            self.w_T = self.w_T + self.step_size * np.conj(e_k) * (z_k / denom)

            self.w = self._to_time_domain(self.w_T)

            self._record_history()
            w_hist_T.append(self.w_T.copy())

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[TDomainLMS] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "coefficients_transform": np.asarray(w_hist_T),
                "power_vector_last": self.power_vector.copy(),
                "transform_matrix": self.T.copy(),
                "assume_unitary": self._assume_unitary,
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF