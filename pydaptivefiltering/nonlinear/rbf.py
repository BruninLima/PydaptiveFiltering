#  nonlinear.rbf.py
#
#       Implements the Radial Basis Function algorithm for REAL valued data.
#       (Algorithm 11.5 - book: Adaptive Filtering: Algorithms and Practical
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
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult
from pydaptivefiltering.utils.validation import ensure_real_signals

ArrayLike = Union[np.ndarray, list]


class RBF(AdaptiveFilter):
    """
    Radial Basis Function (RBF) adaptive model (real-valued).

    Online adaptation of an RBF network with Gaussian basis functions, following
    Diniz (Alg. 11.5). The algorithm updates:
    - output weights ``w`` (one weight per neuron),
    - centers ``c_i`` (stored in ``vet``),
    - spreads ``sigma_i`` (stored in ``sigma``).

    Parameters
    ----------
    n_neurons : int
        Number of RBF neurons (basis functions).
    input_dim : int
        Dimension of the regressor vector ``u[k]``. If :meth:`optimize` is called
        with a 1D input signal, this is interpreted as the tap length.
    ur : float, optional
        Step size for center updates. Default is 1e-2.
    uw : float, optional
        Step size for output-weight updates. Default is 1e-2.
    us : float, optional
        Step size for spread (sigma) updates. Default is 1e-2.
    w_init : array_like of float, optional
        Initial output-weight vector ``w(0)`` with shape ``(n_neurons,)``.
        If None, initializes from a standard normal distribution.
    sigma_init : float, optional
        Initial spread value used for all neurons (must be positive). Default is 1.0.
    centers_init_scale : float, optional
        Scale used for random initialization of centers. Default is 0.5.
    rng : numpy.random.Generator, optional
        Random generator used for reproducible initialization.
    safe_eps : float, optional
        Small positive constant used to guard denominators (e.g., ``sigma^2`` and
        ``sigma^3``). Default is 1e-12.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and parameters
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Model
        For a regressor vector ``u[k] \\in \\mathbb{R}^{D}``, define Gaussian basis
        functions:

        .. math::
            \\phi_i(u[k]) = \\exp\\left(-\\frac{\\|u[k] - c_i\\|^2}{\\sigma_i^2}\\right),

        where ``c_i`` is the center and ``sigma_i > 0`` is the spread of neuron ``i``.
        The network output is

        .. math::
            y[k] = \\sum_{i=1}^{Q} w_i\\, \\phi_i(u[k]) = w^T \\phi(u[k]),

        where ``Q = n_neurons`` and ``\\phi(u[k]) \\in \\mathbb{R}^{Q}`` stacks all
        activations.

    Input formats
        The method :meth:`optimize` accepts two input formats:

        1. **Regressor matrix** ``U`` with shape ``(N, input_dim)``:
           each row is used directly as ``u[k]``.

        2. **Scalar input signal** ``x[k]`` with shape ``(N,)``:
           tapped-delay regressors of length ``input_dim`` are built as

           .. math::
               u[k] = [x[k], x[k-1], \\ldots, x[k-input\\_dim+1]]^T.

    Library conventions
        - ``OptimizationResult.coefficients`` stores the history of the **output
          weights** ``w`` (the neuron output layer).
        - Centers and spreads are returned via ``result.extra`` when
          ``return_internal_states=True``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.5.
    """
    supports_complex: bool = False

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
        centers_init_scale: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        safe_eps: float = 1e-12,
    ) -> None:
        n_neurons = int(n_neurons)
        input_dim = int(input_dim)
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0. Got {n_neurons}.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0. Got {input_dim}.")
        if float(sigma_init) <= 0.0:
            raise ValueError(f"sigma_init must be > 0. Got {sigma_init}.")

        super().__init__(filter_order=n_neurons - 1, w_init=None)

        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.ur = float(ur)
        self.uw = float(uw)
        self.us = float(us)

        self._safe_eps = float(safe_eps)
        self._rng = rng if rng is not None else np.random.default_rng()

        if w_init is None:
            self.w = self._rng.standard_normal(n_neurons).astype(np.float64)
        else:
            w0 = np.asarray(w_init, dtype=np.float64).reshape(-1)
            if w0.size != n_neurons:
                raise ValueError(f"w_init must have length {n_neurons}, got {w0.size}.")
            self.w = w0

        self.vet = (float(centers_init_scale) * self._rng.standard_normal((n_neurons, input_dim))).astype(
            np.float64
        )
        self.sigma = np.ones(n_neurons, dtype=np.float64) * float(sigma_init)

        self.w_history = []
        self._record_history()

    @staticmethod
    def _build_regressors_1d(x: np.ndarray, input_dim: int) -> np.ndarray:
        """Build tapped-delay regressors u[k]=[x[k], x[k-1], ..., x[k-input_dim+1]]."""
        x = np.asarray(x, dtype=np.float64).ravel()
        N = int(x.size)
        m = int(input_dim) - 1
        x_pad = np.zeros(N + m, dtype=np.float64)
        x_pad[m:] = x
        return np.array([x_pad[k : k + m + 1][::-1] for k in range(N)], dtype=np.float64)

    @staticmethod
    def _as_regressor_matrix(x_in: np.ndarray, input_dim: int) -> Tuple[np.ndarray, int]:
        """Return (U, N) from either (N,input_dim) or (N,) input."""
        x_in = np.asarray(x_in, dtype=np.float64)
        if x_in.ndim == 2:
            if x_in.shape[1] != input_dim:
                raise ValueError(f"input_signal must have shape (N,{input_dim}). Got {x_in.shape}.")
            return x_in.astype(np.float64, copy=False), int(x_in.shape[0])
        if x_in.ndim == 1:
            U = RBF._build_regressors_1d(x_in, input_dim=input_dim)
            return U, int(U.shape[0])
        raise ValueError("input_signal must be 1D (signal) or 2D (regressor matrix).")

    @ensure_real_signals
    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the RBF online adaptation loop.

        Parameters
        ----------
        input_signal : array_like of float
            Either:
            - regressor matrix ``U`` with shape ``(N, input_dim)``, or
            - scalar input signal ``x[k]`` with shape ``(N,)`` (tapped-delay
              regressors are built internally).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes final centers/spreads and last activation vector
            in ``result.extra``.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar output sequence ``y[k] = w^T \\phi(u[k])``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Output-weight history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``centers_last`` : ndarray of float
                    Final centers array (shape ``(n_neurons, input_dim)``).
                - ``sigma_last`` : ndarray of float
                    Final spreads vector (shape ``(n_neurons,)``).
                - ``last_phi`` : ndarray of float
                    Last basis-function activation vector ``\\phi(u[k])`` (shape ``(n_neurons,)``).
        """
        t0 = perf_counter()

        x_in = np.asarray(input_signal, dtype=np.float64)
        d_in = np.asarray(desired_signal, dtype=np.float64).ravel()

        U, N = self._as_regressor_matrix(x_in, input_dim=self.input_dim)
        if d_in.size != N:
            raise ValueError(f"Shape mismatch: input({N}) and desired({d_in.size}).")

        outputs = np.zeros(N, dtype=np.float64)
        errors = np.zeros(N, dtype=np.float64)

        last_phi: Optional[np.ndarray] = None

        for k in range(N):
            u = U[k, :]

            diff = u[None, :] - self.vet
            dis_sq = np.sum(diff * diff, axis=1)

            sigma_sq = (self.sigma * self.sigma) + self._safe_eps
            phi = np.exp(-dis_sq / sigma_sq)
            last_phi = phi

            y_k = float(np.dot(self.w, phi))
            outputs[k] = y_k

            e_k = float(d_in[k] - y_k)
            errors[k] = e_k

            self.w = self.w + (2.0 * self.uw) * e_k * phi

            sigma_cu = np.maximum(self.sigma, self._safe_eps)
            self.sigma = self.sigma + (2.0 * self.us) * e_k * phi * self.w * dis_sq / (sigma_cu**3)

            denom_c = (sigma_cu**2) + self._safe_eps
            for p in range(self.n_neurons):
                self.vet[p] = self.vet[p] + (2.0 * self.ur) * phi[p] * e_k * self.w[p] * (u - self.vet[p]) / denom_c[p]

            self._record_history()

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[RBF] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "centers_last": self.vet.copy(),
                "sigma_last": self.sigma.copy(),
                "last_phi": None if last_phi is None else last_phi.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF