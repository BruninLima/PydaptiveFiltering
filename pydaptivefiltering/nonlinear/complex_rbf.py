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
    Complex Radial Basis Function (CRBF) adaptive network (complex-valued).

    Complex-valued RBF adaptive model following Diniz (Alg. 11.6). The network
    output is formed from Gaussian radial basis functions centered at complex
    vectors, combined by complex weights.

    Parameters
    ----------
    n_neurons : int
        Number of RBF neurons (centers/basis functions).
    input_dim : int
        Regressor dimension (length of :math:`u[k]`).
    ur : float, optional
        Step-size for centers update. Default is 0.01.
    uw : float, optional
        Step-size for weights update. Default is 0.01.
    us : float, optional
        Step-size for spread (sigma) update. Default is 0.01.
    w_init : array_like of complex, optional
        Initial neuron weights :math:`w(0)` with shape ``(n_neurons,)``. If None,
        weights are initialized randomly (complex Gaussian).
    sigma_init : float, optional
        Initial spread used for all neurons (must be > 0). Default is 1.0.
    rng : numpy.random.Generator, optional
        Random generator used for reproducible initialization when ``w_init`` is None
        (and for centers initialization). If None, uses ``np.random.default_rng()``.

    Notes
    -----
    Complex-valued
        This implementation supports complex-valued signals and coefficients
        (``supports_complex=True``).

    Input handling
        :meth:`optimize` accepts either:
        1) A 1D input signal ``x[k]`` with shape ``(N,)``. A tapped-delay regressor
           matrix ``U`` with shape ``(N, input_dim)`` is built internally using

           .. math::
              u[k] = [x[k], x[k-1], \\dots, x[k-input\\_dim+1]]^T.

        2) A 2D regressor matrix ``U`` with shape ``(N, input_dim)`` whose rows are
           used directly as :math:`u[k]`.

    RBF activations and output (as implemented)
        For neuron :math:`p` with complex center :math:`c_p \\in \\mathbb{C}^{D}`
        (stored as row ``vet[p, :]``) and real spread :math:`\\sigma_p > 0`
        (stored in ``sigma[p]``), the activation is

        .. math::
            f_p(u[k]) = \\exp\\left( -\\frac{\\lVert u[k] - c_p \\rVert^2}{\\sigma_p^2} \\right),

        where :math:`\\lVert \\cdot \\rVert^2` is implemented as the sum of squared
        real and imaginary parts. Stacking all activations:

        .. math::
            f(u[k]) = [f_1(u[k]), \\dots, f_P(u[k])]^{T} \\in \\mathbb{R}^{P},

        the (a priori) output is computed as

        .. math::
            y[k] = w^H[k-1] f(u[k]) = \\sum_{p=1}^{P} \\overline{w_p[k-1]}\\, f_p(u[k]).

        In code, this corresponds to ``np.vdot(w_old, f)``.

    Adaptation loop (a priori form, as implemented)
        With error

        .. math::
            e[k] = d[k] - y[k],

        the weight update is

        .. math::
            w[k] = w[k-1] + 2\\,\\mu_w\\, \\overline{e[k]}\\, f(u[k]),

        where ``mu_w = uw``. The center and spread updates follow the expressions
        implemented in the code via the intermediate term ``phi = real(e[k] * w_old)``.
        (The exact algebraic form is determined by Alg. 11.6 and the original implementation.)

    Numerical safeguards
        - ``safe_eps`` in :meth:`optimize` guards denominators involving ``sigma``
          to avoid division by very small values.
        - ``sigma`` is clipped from below by ``safe_eps`` after each update.

    Implementation details
        - Coefficient history recorded by the base class corresponds to the neuron
          weights ``w``. Centers (``vet``) and spreads (``sigma``) are not part of the
          base history but can be returned via ``result.extra`` when requested.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.6.
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
        Executes the CRBF adaptation loop over paired regressor/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Either:
            - Input signal ``x[k]`` with shape ``(N,)`` (will be flattened), in which
              case tapped-delay regressors of length ``input_dim`` are built internally; or
            - Regressor matrix ``U`` with shape ``(N, input_dim)`` (each row is ``u[k]``).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"centers_last"``, ``"sigma_last"``, ``"last_activation"``, and
            ``"last_regressor"`` (plus ``"input_dim"`` and ``"n_neurons"``).
        safe_eps : float, optional
            Small positive constant used to guard denominators involving ``sigma``.
            Default is 1e-12.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar a priori output sequence, ``y[k] = w^H[k-1] f(u[k])``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class (neuron weights ``w``).
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
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

        eps = float(safe_eps)

        for k in range(N):
            u = U[k, :]
            last_u = u

            dis_sq = self._squared_distance_complex(u, self.vet)
            sigma_sq = np.maximum(self.sigma**2, float(safe_eps))
            f = np.exp(-dis_sq / sigma_sq)
            last_f = f

            
            w_old = self.w
            y_k = complex(np.vdot(w_old, f))
            outputs[k] = y_k
            e_k = d[k] - y_k 
            errors[k] = e_k

            
            phi = np.real(e_k * w_old)    
            
            denom_sigma = np.maximum(self.sigma**3, eps)
            grad_sigma = (
                (4.0 * self.us)
                * f
                * phi
                * dis_sq
                / denom_sigma
            )
            self.sigma = np.maximum(self.sigma + grad_sigma, eps)

            denom_c = np.maximum(self.sigma**2, eps)

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