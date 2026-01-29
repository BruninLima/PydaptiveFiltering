#  nonlinear.mlp.py
#
#       Implements the Multilayer Perceptron algorithm for REAL valued data.
#       (Algorithm 11.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#       Modified to include Momentum and selectable Activation Functions.
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                            diniz@lps.ufrj.br

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union, Dict, Any, Tuple, List

import numpy as np

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult
from pydaptivefiltering._utils.validation import ensure_real_signals
from pydaptivefiltering._utils.typing import ArrayLike



def _tanh(v: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(v, -50.0, 50.0))


def _dtanh(v: np.ndarray) -> np.ndarray:
    t = np.tanh(np.clip(v, -50.0, 50.0))
    return 1.0 - t * t


def _sigmoid(v: np.ndarray) -> np.ndarray:
    z = np.clip(v, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _dsigmoid(v: np.ndarray) -> np.ndarray:
    s = _sigmoid(v)
    return s * (1.0 - s)


class MultilayerPerceptron(AdaptiveFilter):
    """
    Multilayer Perceptron (MLP) adaptive model with momentum (real-valued).

    Online adaptation of a 2-hidden-layer feedforward neural network using a
    stochastic-gradient update with momentum. The model is treated as an
    adaptive nonlinear filter.

    The forward pass is:

    .. math::
        v_1[k] = W_1 u[k] - b_1, \\qquad y_1[k] = \\phi(v_1[k]),

    .. math::
        v_2[k] = W_2 y_1[k] - b_2, \\qquad y_2[k] = \\phi(v_2[k]),

    .. math::
        y[k] = w_3^T y_2[k] - b_3,

    where ``\\phi`` is either ``tanh`` or ``sigmoid``.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons in each hidden layer. Default is 10.
    input_dim : int, optional
        Dimension of the regressor vector ``u[k]``. Default is 3.
        If :meth:`optimize` is called with a 1D input signal, this must be 3
        (see Notes).
    step_size : float, optional
        Gradient step size ``mu``. Default is 1e-2.
    momentum : float, optional
        Momentum factor in ``[0, 1)``. Default is 0.9.
    activation : {"tanh", "sigmoid"}, optional
        Activation function used in both hidden layers. Default is ``"tanh"``.
    w_init : array_like of float, optional
        Optional initialization for the output-layer weights ``w_3(0)``, with
        shape ``(n_neurons,)``. If None, Xavier/Glorot-style uniform
        initialization is used for all weights.
    rng : numpy.random.Generator, optional
        Random generator used for initialization.

    Notes
    -----
    Real-valued only
        This implementation is restricted to real-valued signals and parameters
        (``supports_complex=False``). The constraint is enforced via
        ``@ensure_real_signals`` on :meth:`optimize`.

    Input formats
        The method :meth:`optimize` accepts two input formats:

        1. **Regressor matrix** ``U`` with shape ``(N, input_dim)``:
           each row is used directly as ``u[k]``.

        2. **Scalar input signal** ``x[k]`` with shape ``(N,)``:
           a 3-dimensional regressor is formed internally as

           .. math::
               u[k] = [x[k],\\ d[k-1],\\ x[k-1]]^T,

           therefore this mode requires ``input_dim = 3``.

    Parameter update (as implemented)
        Let the a priori error be ``e[k] = d[k] - y[k]``. This implementation
        applies a momentum update of the form

        .. math::
            \\theta[k+1] = \\theta[k] + \\Delta\\theta[k] + \\beta\\,\\Delta\\theta[k-1],

        where ``\\beta`` is the momentum factor and ``\\Delta\\theta[k]`` is a
        gradient step proportional to ``e[k]``. (See source for the exact
        per-parameter expressions.)

    Library conventions
        - The base class ``filter_order`` is used only as a size indicator
          (set to ``n_neurons - 1``).
        - ``OptimizationResult.coefficients`` stores a *proxy* coefficient
          history: the output-layer weight vector ``w3`` as tracked through
          ``self.w`` for compatibility with the base API.
        - Full parameter trajectories can be returned in ``result.extra`` when
          ``return_internal_states=True``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 11.4 (MLP adaptive structure; here
       extended with momentum and selectable activations).
    """

    supports_complex: bool = False

    def __init__(
        self,
        n_neurons: int = 10,
        input_dim: int = 3,
        step_size: float = 0.01,
        momentum: float = 0.9,
        activation: str = "tanh",
        w_init: Optional[ArrayLike] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        n_neurons = int(n_neurons)
        input_dim = int(input_dim)
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0. Got {n_neurons}.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0. Got {input_dim}.")
        if not (0.0 <= float(momentum) < 1.0):
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1. Got {momentum}.")

        super().__init__(filter_order=n_neurons - 1, w_init=None)

        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.step_size = float(step_size)
        self.momentum = float(momentum)

        if activation == "tanh":
            self.act_func = _tanh
            self.act_deriv = _dtanh
        elif activation == "sigmoid":
            self.act_func = _sigmoid
            self.act_deriv = _dsigmoid
        else:
            raise ValueError("activation must be 'tanh' or 'sigmoid'.")

        self._rng = rng if rng is not None else np.random.default_rng()

        limit_w1 = float(np.sqrt(6.0 / (input_dim + n_neurons)))
        limit_w2 = float(np.sqrt(6.0 / (n_neurons + n_neurons)))
        limit_w3 = float(np.sqrt(6.0 / (n_neurons + 1)))

        self.w1 = self._rng.uniform(-limit_w1, limit_w1, (n_neurons, input_dim)).astype(np.float64)
        self.w2 = self._rng.uniform(-limit_w2, limit_w2, (n_neurons, n_neurons)).astype(np.float64)
        self.w3 = self._rng.uniform(-limit_w3, limit_w3, (n_neurons,)).astype(np.float64)

        if w_init is not None:
            w3_0 = np.asarray(w_init, dtype=np.float64).reshape(-1)
            if w3_0.size != n_neurons:
                raise ValueError(f"w_init must have length {n_neurons}, got {w3_0.size}.")
            self.w3 = w3_0

        self.b1 = np.zeros(n_neurons, dtype=np.float64)
        self.b2 = np.zeros(n_neurons, dtype=np.float64)
        self.b3 = 0.0

        self.prev_dw1 = np.zeros_like(self.w1)
        self.prev_dw2 = np.zeros_like(self.w2)
        self.prev_dw3 = np.zeros_like(self.w3)
        self.prev_db1 = np.zeros_like(self.b1)
        self.prev_db2 = np.zeros_like(self.b2)
        self.prev_db3 = 0.0

        self.w = self.w3.copy()
        self.w_history = []
        self._record_history()

    @staticmethod
    def _as_regressor_matrix(
        x_in: np.ndarray, d_in: np.ndarray, input_dim: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Return (U, is_multidim).

        - If x_in is 2D: U = x_in
        - If x_in is 1D: builds U[k]=[x[k], d[k-1], x[k-1]] and requires input_dim=3
        """
        x_in = np.asarray(x_in, dtype=np.float64)
        d_in = np.asarray(d_in, dtype=np.float64).ravel()

        if x_in.ndim == 2:
            if x_in.shape[0] != d_in.size:
                raise ValueError(f"Shape mismatch: input({x_in.shape[0]}) and desired({d_in.size}).")
            if x_in.shape[1] != input_dim:
                raise ValueError(f"input_signal second dim must be input_dim={input_dim}. Got {x_in.shape}.")
            return x_in.astype(np.float64, copy=False), True

        if x_in.ndim == 1:
            if input_dim != 3:
                raise ValueError(
                    "When input_signal is 1D, this implementation uses u[k]=[x[k], d[k-1], x[k-1]] "
                    "so input_dim must be 3."
                )
            if x_in.size != d_in.size:
                raise ValueError(f"Shape mismatch: input({x_in.size}) and desired({d_in.size}).")

            N = int(x_in.size)
            U = np.zeros((N, 3), dtype=np.float64)
            x_prev = 0.0
            d_prev = 0.0
            for k in range(N):
                U[k, :] = np.array([x_in[k], d_prev, x_prev], dtype=np.float64)
                x_prev = float(x_in[k])
                d_prev = float(d_in[k])
            return U, False

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
        Executes the online MLP adaptation loop (with momentum).

        Parameters
        ----------
        input_signal : array_like of float
            Either:
            - regressor matrix ``U`` with shape ``(N, input_dim)``, or
            - scalar input signal ``x[k]`` with shape ``(N,)`` (in which case the
              regressor is built as ``u[k] = [x[k], d[k-1], x[k-1]]`` and
              requires ``input_dim = 3``).
        desired_signal : array_like of float
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, stores parameter snapshots in ``result.extra`` (may be memory
            intensive for long runs).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of float, shape ``(N,)``
                Scalar output sequence ``y[k]``.
            - errors : ndarray of float, shape ``(N,)``
                Scalar a priori error sequence, ``e[k] = d[k] - y[k]``.
            - coefficients : ndarray of float
                Proxy coefficient history recorded by the base class (tracks
                the output-layer weights ``w3``).
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` with:
                - ``w1_hist`` : list of ndarray
                    Hidden-layer-1 weight snapshots.
                - ``w2_hist`` : list of ndarray
                    Hidden-layer-2 weight snapshots.
                - ``w3_hist`` : list of ndarray
                    Output-layer weight snapshots.
                - ``b1_hist`` : list of ndarray
                    Bias-1 snapshots.
                - ``b2_hist`` : list of ndarray
                    Bias-2 snapshots.
                - ``b3_hist`` : list of float
                    Bias-3 snapshots.
                - ``activation`` : str
                    Activation identifier (``"tanh"`` or ``"sigmoid"``).
        """
        t0 = perf_counter()

        x_in = np.asarray(input_signal, dtype=np.float64)
        d_in = np.asarray(desired_signal, dtype=np.float64).ravel()

        U, _ = self._as_regressor_matrix(x_in, d_in, self.input_dim)
        N = int(U.shape[0])

        outputs = np.zeros(N, dtype=np.float64)
        errors = np.zeros(N, dtype=np.float64)

        w1_hist: List[np.ndarray] = []
        w2_hist: List[np.ndarray] = []
        w3_hist: List[np.ndarray] = []
        b1_hist: List[np.ndarray] = []
        b2_hist: List[np.ndarray] = []
        b3_hist: List[float] = []

        for k in range(N):
            u = U[k, :]

            v1 = (self.w1 @ u) - self.b1
            y1 = self.act_func(v1)

            v2 = (self.w2 @ y1) - self.b2
            y2 = self.act_func(v2)

            y_k = float(np.dot(y2, self.w3) - self.b3)
            outputs[k] = y_k
            e_k = float(d_in[k] - y_k)
            errors[k] = e_k

            er_hid2 = e_k * self.w3 * self.act_deriv(v2)
            er_hid1 = (self.w2.T @ er_hid2) * self.act_deriv(v1)

            dw3 = (2.0 * self.step_size) * e_k * y2
            self.w3 = self.w3 + dw3 + self.momentum * self.prev_dw3
            self.prev_dw3 = dw3

            db3 = (-2.0 * self.step_size) * e_k
            self.b3 = float(self.b3 + db3 + self.momentum * self.prev_db3)
            self.prev_db3 = db3

            dw2 = (2.0 * self.step_size) * np.outer(er_hid2, y1)
            self.w2 = self.w2 + dw2 + self.momentum * self.prev_dw2
            self.prev_dw2 = dw2

            db2 = (-2.0 * self.step_size) * er_hid2
            self.b2 = self.b2 + db2 + self.momentum * self.prev_db2
            self.prev_db2 = db2

            dw1 = (2.0 * self.step_size) * np.outer(er_hid1, u)
            self.w1 = self.w1 + dw1 + self.momentum * self.prev_dw1
            self.prev_dw1 = dw1

            db1 = (-2.0 * self.step_size) * er_hid1
            self.b1 = self.b1 + db1 + self.momentum * self.prev_db1
            self.prev_db1 = db1

            self.w = self.w3.copy()
            self._record_history()

            if return_internal_states:
                w1_hist.append(self.w1.copy())
                w2_hist.append(self.w2.copy())
                w3_hist.append(self.w3.copy())
                b1_hist.append(self.b1.copy())
                b2_hist.append(self.b2.copy())
                b3_hist.append(float(self.b3))

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[MultilayerPerceptron] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "w1_hist": w1_hist,
                "w2_hist": w2_hist,
                "w3_hist": w3_hist,
                "b1_hist": b1_hist,
                "b2_hist": b2_hist,
                "b3_hist": b3_hist,
                "activation": "tanh" if self.act_func is _tanh else "sigmoid",
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF