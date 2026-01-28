#  nonlinear.complex_rbf.py
#
#       Implements the Complex Radial Basis Function algorithm for COMPLEX valued data.
#       (Algorithm 11.6 - book: Adaptive Filtering: Algorithms and Practical
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
from typing import Optional, Union, Dict, Any

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult

ArrayLike = Union[np.ndarray, list]


class ComplexRBF(AdaptiveFilter):
    """
    Complex Radial Basis Function (CRBF) network (complex-valued).

    Implements a complex-valued RBF adaptive model (Algorithm 11.6 - Diniz).
    The model output is computed as:

        f_p(u) = exp( -||u - c_p||^2 / sigma_p^2 )
        y[k]   = w^H f(u_k)

    where:
      - u_k is the input regressor (dimension = input_dim),
      - c_p are complex centers ("vet" in the original code),
      - sigma_p are real spreads,
      - w are complex neuron weights.

    Input handling
    --------------
    This implementation accepts two input formats in `optimize`:

    1) 1D input signal x[k] (shape (N,)):
       A tapped-delay regressor u_k of length `input_dim` is formed internally.

    2) 2D regressor matrix U (shape (N, input_dim)):
       Each row is used directly as u_k.

    Notes
    -----
    - Complex-valued implementation (`supports_complex=True`).
    - The base class `filter_order` is used here as a size indicator (n_neurons-1).
    - `OptimizationResult.coefficients` stores the history of neuron weights `w`.
      Centers and spreads can be returned via `result.extra` when requested.
    """

    supports_complex: bool = True

    def __init__(
        self,
        n_neurons: int,
        input_dim: int,
        ur: float = 0.01,
        uw: float = 0.01,
        us: float = 0.01,
        w_init: Optional[ArrayLike] = None,
        *,
        sigma_init: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_neurons:
            Number of RBF neurons.
        input_dim:
            Dimension of the input regressor u_k.
        ur:
            Step-size for centers update.
        uw:
            Step-size for weights update.
        us:
            Step-size for spread (sigma) update.
        w_init:
            Optional initial neuron weights (length n_neurons). If None, random complex.
        sigma_init:
            Initial spread value used for all neurons (must be > 0).
        rng:
            Optional numpy random generator for reproducible initialization.
        """
        n_neurons = int(n_neurons)
        input_dim = int(input_dim)
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0. Got {n_neurons}.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0. Got {input_dim}.")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0. Got {sigma_init}.")

        super().__init__(filter_order=n_neurons - 1, w_init=None)

        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.ur = float(ur)
        self.uw = float(uw)
        self.us = float(us)

        self._rng = rng if rng is not None else np.random.default_rng()

        if w_init is None:
            w0 = self._rng.standard_normal(n_neurons) + 1j * self._rng.standard_normal(n_neurons)
            self.w = w0.astype(complex)
        else:
            w0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if w0.size != n_neurons:
                raise ValueError(f"w_init must have length {n_neurons}, got {w0.size}.")
            self.w = w0

        self.vet = 0.5 * (
            self._rng.standard_normal((n_neurons, input_dim))
            + 1j * self._rng.standard_normal((n_neurons, input_dim))
        ).astype(complex)

        self.sigma = np.ones(n_neurons, dtype=float) * float(sigma_init)

        self.w_history = []
        self._record_history()

    @staticmethod
    def _build_regressors_from_signal(x: np.ndarray, input_dim: int) -> np.ndarray:
        """Build tapped-delay regressors from a 1D signal (N,)->(N,input_dim)."""
        x = np.asarray(x, dtype=complex).ravel()
        n = int(x.size)
        m = int(input_dim - 1)

        x_padded = np.zeros(n + m, dtype=complex)
        x_padded[m:] = x

        U = np.zeros((n, input_dim), dtype=complex)
        for k in range(n):
            U[k, :] = x_padded[k : k + input_dim][::-1]
        return U

    @staticmethod
    def _squared_distance_complex(u: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute ||u - c_p||^2 for each center row.
        u: (input_dim,)
        centers: (n_neurons, input_dim)
        returns: (n_neurons,)
        """
        diff = u[None, :] - centers
        return np.sum(diff.real**2 + diff.imag**2, axis=1)

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
        return_internal_states: bool = False,
        *,
        safe_eps: float = 1e-12,
    ) -> OptimizationResult:
        """
        Run CRBF adaptation.

        Parameters
        ----------
        input_signal:
            Either:
              - 1D signal x[k] with shape (N,), or
              - regressor matrix U with shape (N, input_dim).
        desired_signal:
            Desired signal d[k], shape (N,).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns final centers/spreads and last activation vector in result.extra.
        safe_eps:
            Small epsilon to protect denominators (sigma and other divisions).

        Returns
        -------
        OptimizationResult
            outputs:
                Model output y[k].
            errors:
                A priori error e[k] = d[k] - y[k].
            coefficients:
                History of neuron weights w[k] (shape (N+1, n_neurons) in base history).
            error_type:
                "a_priori".

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["centers_last"]:
            Final centers array (n_neurons, input_dim).
        extra["sigma_last"]:
            Final spreads array (n_neurons,).
        extra["last_activation"]:
            Last activation vector f(u_k) (n_neurons,).
        extra["last_regressor"]:
            Last regressor u_k (input_dim,).
        """
        t0 = perf_counter()

        x_in = np.asarray(input_signal)
        d = np.asarray(desired_signal, dtype=complex).ravel()

        if x_in.ndim == 1:
            U = self._build_regressors_from_signal(x_in, self.input_dim)
        elif x_in.ndim == 2:
            U = np.asarray(x_in, dtype=complex)
            if U.shape[1] != self.input_dim:
                raise ValueError(
                    f"input_signal has shape {U.shape}, expected second dim input_dim={self.input_dim}."
                )
        else:
            raise ValueError("input_signal must be 1D (signal) or 2D (regressor matrix).")

        N = int(U.shape[0])
        if d.size != N:
            raise ValueError(f"Inconsistent lengths: regressors({N}) != desired({d.size}).")

        outputs = np.zeros(N, dtype=complex)
        errors  = np.zeros(N, dtype=complex)

        last_f: Optional[np.ndarray] = None
        last_u: Optional[np.ndarray] = None

        for k in range(N):
            u = U[k, :]
            last_u = u

            dis_sq = self._squared_distance_complex(u, self.vet)
            sigma_sq = np.maximum(self.sigma**2, float(safe_eps))
            f = np.exp(-dis_sq / sigma_sq)
            last_f = f

            
            w_old = self.w.copy()
            y_k = complex(np.vdot(w_old, f))
            outputs[k] = y_k
            e_k = d[k] - y_k 
            errors[k] = e_k

            self.w = w_old + (2.0 * self.uw) * np.conj(e_k) * f
            phi = np.real(e_k * w_old)    
            denom_sigma = np.maximum(self.sigma**3, float(safe_eps))
            grad_sigma = (
                (4.0 * self.us)
                * f
                * phi
                * dis_sq
                / denom_sigma
            )
            self.sigma = np.maximum(self.sigma + grad_sigma, float(safe_eps))

            denom_c = np.maximum(self.sigma**2, safe_eps)

            self.vet = self.vet + (2.0 * self.ur) * (f[:, None] * phi[:, None]) * (u - self.vet) / denom_c[:, None]

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[ComplexRBF] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "centers_last": self.vet.copy(),
                "sigma_last": self.sigma.copy(),
                "last_activation": None if last_f is None else np.asarray(last_f).copy(),
                "last_regressor": None if last_u is None else np.asarray(last_u).copy(),
                "input_dim": int(self.input_dim),
                "n_neurons": int(self.n_neurons),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF