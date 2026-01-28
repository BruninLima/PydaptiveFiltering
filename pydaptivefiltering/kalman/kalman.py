# kalman.kalman.py
#
#       Implements the Kalman Filter algorithm for COMPLEX or REAL valued data.
#       (Algorithm 17.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho  - cpneqs@gmail.com          & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                           diniz@lps.ufrj.br
#

from __future__ import annotations

import numpy as np
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult

ArrayLike = Union[np.ndarray, list]


def _as_2d_col(x: np.ndarray) -> np.ndarray:
    """Force an input vector into shape (n, 1).

    Accepts
    -------
    x:
        Array-like compatible with:
        - (n,)
        - (n,1)
        - (1,n)

    Returns
    -------
    np.ndarray
        Column vector with shape (n, 1).

    Raises
    ------
    ValueError
        If x cannot be interpreted as a vector.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim == 2 and x.shape[0] == 1:
        return x.reshape(-1, 1)
    raise ValueError(f"Expected a vector compatible with (n,1). Got shape={x.shape}.")


def _as_meas_matrix(y_seq: np.ndarray) -> np.ndarray:
    """Normalize a measurement sequence to a 2-D matrix (N, p).

    Accepts y as:
      - (N,)       -> (N, 1)   (scalar measurements)
      - (N, p)     -> (N, p)
      - (N, p, 1)  -> (N, p)   (squeezes last singleton axis)

    Parameters
    ----------
    y_seq:
        Measurement sequence.

    Returns
    -------
    np.ndarray
        Measurement matrix with shape (N, p).

    Raises
    ------
    ValueError
        If y_seq has an unsupported shape.
    """
    y_seq = np.asarray(y_seq)
    if y_seq.ndim == 1:
        return y_seq.reshape(-1, 1)
    if y_seq.ndim == 2:
        return y_seq
    if y_seq.ndim == 3 and y_seq.shape[-1] == 1:
        return y_seq[..., 0]
    raise ValueError(
        f"input_signal must have shape (N,), (N,p) or (N,p,1). Got {y_seq.shape}."
    )


def _mat_at_k(mat_or_seq: Union[np.ndarray, Sequence[np.ndarray]], k: int) -> np.ndarray:
    """Return a matrix for iteration k (time-varying or constant).

    Parameters
    ----------
    mat_or_seq:
        Either a constant numpy array or a (list/tuple) sequence of arrays over k.
    k:
        Iteration index.

    Returns
    -------
    np.ndarray
        The matrix to be used at time k.
    """
    if isinstance(mat_or_seq, (list, tuple)):
        return np.asarray(mat_or_seq[k])
    return np.asarray(mat_or_seq)


class Kalman(AdaptiveFilter):
    """
    Kalman filter for state estimation (real or complex-valued).

    Implements the discrete-time Kalman filter recursion for linear state-space
    models with additive process and measurement noise. Matrices may be constant
    (single ``ndarray``) or time-varying (a sequence of arrays indexed by ``k``).

    The model used is:

    .. math::
        x(k) = A(k-1) x(k-1) + B(k) n(k),

    .. math::
        y(k) = C^T(k) x(k) + n_1(k),

    where :math:`n(k)` is the process noise with covariance :math:`R_n(k)` and
    :math:`n_1(k)` is the measurement noise with covariance :math:`R_{n1}(k)`.

    Notes
    -----
    API integration
    ~~~~~~~~~~~~~~~
    This class inherits from :class:`~pydaptivefiltering.base.AdaptiveFilter` to
    share a common interface. Here, the "weights" are the state estimate:
    ``self.w`` stores the current state vector (flattened), and
    ``self.w_history`` stores the covariance matrices over time.

    Time-varying matrices
    ~~~~~~~~~~~~~~~~~~~~~
    Any of ``A``, ``C_T``, ``B``, ``Rn``, ``Rn1`` may be provided either as:
    - a constant ``ndarray``, used for all k; or
    - a sequence (list/tuple) of ``ndarray``, where element ``k`` is used at time k.

    Dimensions
    ~~~~~~~~~~
    Let ``n`` be the state dimension, ``p`` the measurement dimension, and ``q``
    the process-noise dimension. Then:

    - ``A(k)`` has shape ``(n, n)``
    - ``C_T(k)`` has shape ``(p, n)``  (note: this is :math:`C^T`)
    - ``B(k)`` has shape ``(n, q)``
    - ``Rn(k)`` has shape ``(q, q)``
    - ``Rn1(k)`` has shape ``(p, p)``

    If ``B`` is not provided, the implementation uses ``B = I`` (thus ``q = n``),
    and expects ``Rn`` to be shape ``(n, n)``.

    Parameters
    ----------
    A : ndarray or Sequence[ndarray]
        State transition matrix :math:`A(k-1)` with shape ``(n, n)``.
    C_T : ndarray or Sequence[ndarray]
        Measurement matrix :math:`C^T(k)` with shape ``(p, n)``.
    Rn : ndarray or Sequence[ndarray]
        Process noise covariance :math:`R_n(k)` with shape ``(q, q)``.
    Rn1 : ndarray or Sequence[ndarray]
        Measurement noise covariance :math:`R_{n1}(k)` with shape ``(p, p)``.
    B : ndarray or Sequence[ndarray], optional
        Process noise input matrix :math:`B(k)` with shape ``(n, q)``.
        If None, uses identity.
    x_init : ndarray, optional
        Initial state estimate :math:`x(0|0)`. Accepts shapes compatible with
        ``(n,)``, ``(n,1)``, or ``(1,n)``. If None, initializes with zeros.
    Re_init : ndarray, optional
        Initial estimation error covariance :math:`R_e(0|0)` with shape ``(n, n)``.
        If None, initializes with identity.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 17.1.
    """
    supports_complex: bool = True

    A: Union[np.ndarray, Sequence[np.ndarray]]
    C_T: Union[np.ndarray, Sequence[np.ndarray]]
    Rn: Union[np.ndarray, Sequence[np.ndarray]]
    Rn1: Union[np.ndarray, Sequence[np.ndarray]]
    B: Optional[Union[np.ndarray, Sequence[np.ndarray]]]

    x: np.ndarray
    Re: np.ndarray

    def __init__(
        self,
        A: Union[np.ndarray, Sequence[np.ndarray]],
        C_T: Union[np.ndarray, Sequence[np.ndarray]],
        Rn: Union[np.ndarray, Sequence[np.ndarray]],
        Rn1: Union[np.ndarray, Sequence[np.ndarray]],
        B: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        x_init: Optional[np.ndarray] = None,
        Re_init: Optional[np.ndarray] = None,
    ) -> None:
        A0 = _mat_at_k(A, 0)
        if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
            raise ValueError(f"A must be square (n,n). Got {A0.shape}.")
        n = int(A0.shape[0])

        super().__init__(filter_order=n - 1, w_init=None)

        self.A = A
        self.C_T = C_T
        self.Rn = Rn
        self.Rn1 = Rn1
        self.B = B

        dtype = np.result_type(
            A0, _mat_at_k(C_T, 0), _mat_at_k(Rn, 0), _mat_at_k(Rn1, 0)
        )
        dtype = np.float64 if np.issubdtype(dtype, np.floating) else np.complex128

        self._dtype = dtype
        self.regressor = np.zeros(self.filter_order + 1, dtype=self._dtype)
        self.w = np.zeros(self.filter_order + 1, dtype=self._dtype)

        if x_init is None:
            x0 = np.zeros((n, 1), dtype=dtype)
        else:
            x0 = _as_2d_col(np.asarray(x_init, dtype=dtype))
            if x0.shape[0] != n:
                raise ValueError(f"x_init must have length n={n}. Got {x0.shape}.")
        self.x = x0

        if Re_init is None:
            Re0 = np.eye(n, dtype=dtype)
        else:
            Re0 = np.asarray(Re_init, dtype=dtype)
            if Re0.shape != (n, n):
                raise ValueError(f"Re_init must be shape (n,n)={(n,n)}. Got {Re0.shape}.")
        self.Re = Re0

        self.w = self.x[:, 0].copy()

        self.w_history = []

    def _validate_step_shapes(
        self,
        A: np.ndarray,
        C_T: np.ndarray,
        Rn: np.ndarray,
        Rn1: np.ndarray,
        B: np.ndarray,
    ) -> None:
        """Validate per-iteration matrix shapes.

        Raises
        ------
        ValueError
            If any matrix has an unexpected shape for the current state dimension.
        """
        n = int(self.x.shape[0])
        if A.shape != (n, n):
            raise ValueError(f"A(k) must be {(n,n)}. Got {A.shape}.")
        if C_T.ndim != 2 or C_T.shape[1] != n:
            raise ValueError(f"C_T(k) must be (p,n) with n={n}. Got {C_T.shape}.")
        p = int(C_T.shape[0])
        if Rn1.shape != (p, p):
            raise ValueError(f"Rn1(k) must be {(p,p)}. Got {Rn1.shape}.")
        if B.ndim != 2 or B.shape[0] != n:
            raise ValueError(f"B(k) must be (n,q) with n={n}. Got {B.shape}.")
        q = int(B.shape[1])
        if Rn.shape != (q, q):
            raise ValueError(f"Rn(k) must be {(q,q)}. Got {Rn.shape}.")

    def optimize(
        self,
        input_signal: ArrayLike,
        desired_signal: Optional[ArrayLike] = None,
        verbose: bool = False,
        return_internal_states: bool = False,
        safe_eps: float = 1e-12,
    ) -> OptimizationResult:
        """
        Executes the Kalman recursion for a sequence of measurements ``y[k]``.

        Parameters
        ----------
        input_signal : array_like
            Measurement sequence ``y[k]``. Accepted shapes:
            - ``(N,)``       for scalar measurements
            - ``(N, p)``     for p-dimensional measurements
            - ``(N, p, 1)``  also accepted (squeezed to ``(N, p)``)
        desired_signal : array_like, optional
            Ignored (kept only for API standardization).
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, returns selected internal values in ``result.extra``.
        safe_eps : float, optional
            Small positive value used to regularize the innovation covariance
            matrix if a linear solve fails (numerical stabilization).

        Returns
        -------
        OptimizationResult
            outputs : ndarray
                State estimates ``x(k|k)``, shape ``(N, n)``.
            errors : ndarray
                Innovations ``v(k) = y(k) - C^T(k) x(k|k-1)``, shape ``(N, p)``.
            coefficients : ndarray
                Covariance history ``R_e(k|k)``, shape ``(N, n, n)``.
            error_type : str
                ``"innovation"``.
            extra : dict, optional
                Present only if ``return_internal_states=True``. See below.

        Extra (when return_internal_states=True)
        --------------------------------------
        kalman_gain_last : ndarray
            Kalman gain ``K`` at the last iteration, shape ``(n, p)``.
        predicted_state_last : ndarray
            Predicted state ``x(k|k-1)`` at the last iteration, shape ``(n,)``.
        predicted_cov_last : ndarray
            Predicted covariance ``R_e(k|k-1)`` at the last iteration, shape ``(n, n)``.
        innovation_cov_last : ndarray
            Innovation covariance ``S`` at the last iteration, shape ``(p, p)``.
        safe_eps : float
            The stabilization epsilon used when regularizing ``S``.
        """
        t0 = perf_counter()

        y_mat = _as_meas_matrix(np.asarray(input_signal))
        y_mat = y_mat.astype(self._dtype, copy=False)

        N = int(y_mat.shape[0])
        n = int(self.x.shape[0])
        p_dim = int(y_mat.shape[1])

        outputs = np.zeros((N, n), dtype=self._dtype)
        errors = np.zeros((N, p_dim), dtype=self._dtype)

        I_n = np.eye(n, dtype=self._dtype)

        self.w_history = []

        last_K: Optional[np.ndarray] = None
        last_x_pred: Optional[np.ndarray] = None
        last_Re_pred: Optional[np.ndarray] = None
        last_S: Optional[np.ndarray] = None

        for k in range(N):
            A_k = np.asarray(_mat_at_k(self.A, k), dtype=self._dtype)
            C_T_k = np.asarray(_mat_at_k(self.C_T, k), dtype=self._dtype)
            Rn_k = np.asarray(_mat_at_k(self.Rn, k), dtype=self._dtype)
            Rn1_k = np.asarray(_mat_at_k(self.Rn1, k), dtype=self._dtype)
            B_k = np.asarray(
                _mat_at_k(self.B, k) if self.B is not None else I_n,
                dtype=self._dtype,
            )

            self._validate_step_shapes(A_k, C_T_k, Rn_k, Rn1_k, B_k)

            y_k = _as_2d_col(y_mat[k]).astype(self._dtype, copy=False)
            C_k = C_T_k.conj().T

            x_pred = A_k @ self.x
            Re_pred = (A_k @ self.Re @ A_k.conj().T) + (B_k @ Rn_k @ B_k.conj().T)

            e_k = y_k - (C_T_k @ x_pred)

            S = (C_T_k @ Re_pred @ C_k) + Rn1_k

            RC = Re_pred @ C_k
            try:
                K = np.linalg.solve(S.conj().T, RC.conj().T).conj().T
            except np.linalg.LinAlgError:
                S_reg = S + (safe_eps * np.eye(p_dim, dtype=self._dtype))
                K = np.linalg.solve(S_reg.conj().T, RC.conj().T).conj().T

            self.x = x_pred + (K @ e_k)
            self.Re = (I_n - (K @ C_T_k)) @ Re_pred

            outputs[k, :] = self.x[:, 0]
            errors[k, :] = e_k[:, 0]

            self.w_history.append(self.Re.copy())

            self.w = self.x[:, 0].copy()

            last_K = K
            last_x_pred = x_pred
            last_Re_pred = Re_pred
            last_S = S

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[Kalman] Completed in {runtime_s * 1000:.03f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "kalman_gain_last": last_K,
                "predicted_state_last": None if last_x_pred is None else last_x_pred[:, 0].copy(),
                "predicted_cov_last": last_Re_pred,
                "innovation_cov_last": last_S,
                "safe_eps": float(safe_eps),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="innovation",
            extra=extra,
        )
# EOF