#  blind.kalman.py
#
#       Implements the Kalman Filter algorithm for COMPLEX or REAL valued data.
#       (Algorithm 17.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#       Authors:
#        . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                           diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Optional, Union, List, Dict, Any, Sequence

from pydaptivefiltering.base import AdaptiveFilter

ArrayLike = Union[np.ndarray, list]


def _as_2d_col(x: np.ndarray) -> np.ndarray:
    """Force vector into shape (n, 1)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim == 2 and x.shape[0] == 1:
        return x.reshape(-1, 1)
    raise ValueError(f"Expected a vector compatible with (n,1). Got shape={x.shape}.")


def _as_meas_matrix(y_seq: np.ndarray) -> np.ndarray:
    """
    Accept y as:
      - (N,) -> (N,1)
      - (N,p) -> (N,p)
      - (N,p,1) -> (N,p)
    """
    y_seq = np.asarray(y_seq)
    if y_seq.ndim == 1:
        return y_seq.reshape(-1, 1)
    if y_seq.ndim == 2:
        return y_seq
    if y_seq.ndim == 3 and y_seq.shape[-1] == 1:
        return y_seq[..., 0]
    raise ValueError(f"input_signal must have shape (N,), (N,p) or (N,p,1). Got {y_seq.shape}.")


def _mat_at_k(mat_or_seq: Union[np.ndarray, Sequence[np.ndarray]], k: int) -> np.ndarray:
    """Return matrix for iteration k (constant matrix or sequence)."""
    if isinstance(mat_or_seq, (list, tuple)):
        return np.asarray(mat_or_seq[k])
    return np.asarray(mat_or_seq)


class Kalman(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Kalman Filter for state estimation with complex or real valued data.
        (Algorithm 17.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

        State-space model (standard form):
            x(k) = A(k-1) x(k-1) + B(k) n(k)
            y(k) = C^T(k) x(k) + n1(k)

        with covariances:
            E[n(k) n(k)^H]   = Rn(k)
            E[n1(k)n1(k)^H]  = Rn1(k)

    Notes
    -----
        - This class inherits from AdaptiveFilter for consistency with the library API.
        - Here, `self.w` is used to store the *current state estimate* x(k|k) as a 1-D vector.
        - `coefficients` history returned by `optimize` stores the error covariance matrices Re(k|k).

    Attributes
    ----------
        supports_complex : bool
            True (Supports complex-valued data).
    """

    supports_complex: bool = True

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
        """
        Inputs
        -------
            A : np.ndarray | sequence[np.ndarray]
                State transition matrix A(k-1), shape (n,n), or a sequence over k.
            C_T : np.ndarray | sequence[np.ndarray]
                Measurement matrix C^T(k), shape (p,n), or a sequence over k.
            Rn : np.ndarray | sequence[np.ndarray]
                Process noise covariance Rn(k), shape (q,q) (q matches B columns), or sequence.
            Rn1 : np.ndarray | sequence[np.ndarray]
                Measurement noise covariance Rn1(k), shape (p,p), or sequence.
            B : np.ndarray | sequence[np.ndarray] | None
                Process noise input matrix B(k), shape (n,q). If None, uses Identity (q=n).
            x_init : np.ndarray | None
                Initial state estimate x(0|0), shape (n,) or (n,1). If None, zeros.
            Re_init : np.ndarray | None
                Initial error covariance Re(0|0), shape (n,n). If None, Identity.

        Raises
        ------
            ValueError
                If shapes are inconsistent.
        """
        A0 = _mat_at_k(A, 0)
        if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
            raise ValueError(f"A must be square (n,n). Got {A0.shape}.")
        n = int(A0.shape[0])

        # Initialize AdaptiveFilter with "order" = n-1 and store x in self.w
        super().__init__(m=n - 1, w_init=None)

        self.A = A
        self.C_T = C_T
        self.Rn = Rn
        self.Rn1 = Rn1
        self.B = B

        dtype = np.result_type(A0, _mat_at_k(C_T, 0), _mat_at_k(Rn, 0), _mat_at_k(Rn1, 0))
        if np.issubdtype(dtype, np.floating):
            dtype = np.float64
        else:
            dtype = np.complex128

        # x(0|0)
        if x_init is None:
            x0 = np.zeros((n, 1), dtype=dtype)
        else:
            x0 = _as_2d_col(np.asarray(x_init, dtype=dtype))
            if x0.shape[0] != n:
                raise ValueError(f"x_init must have length n={n}. Got {x0.shape}.")
        self.x = x0

        # Re(0|0)
        if Re_init is None:
            Re0 = np.eye(n, dtype=dtype)
        else:
            Re0 = np.asarray(Re_init, dtype=dtype)
            if Re0.shape != (n, n):
                raise ValueError(f"Re_init must be shape (n,n)={(n,n)}. Got {Re0.shape}.")
        self.Re = Re0

        # Keep self.w as 1-D view of current state for consistency
        self.w = self.x[:, 0].copy()

        self.Re_history: List[np.ndarray] = []

    def _validate_step_shapes(self, A: np.ndarray, C_T: np.ndarray, Rn: np.ndarray, Rn1: np.ndarray, B: np.ndarray) -> None:
        n = self.x.shape[0]
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
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the Kalman filtering recursion for a sequence of measurements y(k).

        Inputs
        -------
            input_signal : np.ndarray | list
                Measurement sequence y(k). Accepted shapes:
                  - (N,)        for scalar measurements
                  - (N,p)       for p-dimensional measurements
                  - (N,p,1)     also accepted (will be squeezed)
            verbose : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs : np.ndarray
                    State estimates x(k|k) with shape (N, n_states).
                errors : np.ndarray
                    Innovations e(k)=y(k)-C^T(k)x(k|k-1) with shape (N, n_meas).
                coefficients : list[np.ndarray]
                    Error covariance matrices Re(k|k), list of shape (n_states, n_states).

        Authors
        -------
            . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                           diniz@lps.ufrj.br
        """
        tic = time()

        y_mat = _as_meas_matrix(np.asarray(input_signal))
        N = int(y_mat.shape[0])

        n = int(self.x.shape[0])
        x_out = np.zeros((N, n), dtype=self.x.dtype)
        e_out = np.zeros((N, y_mat.shape[1]), dtype=self.x.dtype)

        self.Re_history = []
        I = np.eye(n, dtype=self.x.dtype)

        for k in range(N):
            # Get step matrices (constant or time-varying)
            A = _mat_at_k(self.A, k)
            C_T = _mat_at_k(self.C_T, k)
            Rn = _mat_at_k(self.Rn, k)
            Rn1 = _mat_at_k(self.Rn1, k)

            if self.B is None:
                B = np.eye(n, dtype=self.x.dtype)
            else:
                B = _mat_at_k(self.B, k)

            A = np.asarray(A, dtype=self.x.dtype)
            C_T = np.asarray(C_T, dtype=self.x.dtype)
            Rn = np.asarray(Rn, dtype=self.x.dtype)
            Rn1 = np.asarray(Rn1, dtype=self.x.dtype)
            B = np.asarray(B, dtype=self.x.dtype)

            self._validate_step_shapes(A, C_T, Rn, Rn1, B)

            y_k = _as_2d_col(y_mat[k]).astype(self.x.dtype)   # (p,1)
            C = C_T.conj().T                                   # (n,p)

            # --- Prediction ---
            # x(k|k-1)
            x_pred = A @ self.x
            # Re(k|k-1)
            Re_pred = (A @ self.Re @ A.conj().T) + (B @ Rn @ B.conj().T)

            # --- Gain ---
            # S = C^T Re C + Rn1  (shape p x p)
            S = (C_T @ Re_pred @ C) + Rn1

            # K = Re_pred C S^{-1}  (solve avoids explicit inverse)
            # Solve S^T Z^T = (Re_pred C)^T for Z = K
            RC = Re_pred @ C  # (n,p)
            K = np.linalg.solve(S.conj().T, RC.conj().T).conj().T  # (n,p)

            # --- Innovation ---
            e_k = y_k - (C_T @ x_pred)  # (p,1)

            # --- Update ---
            self.x = x_pred + (K @ e_k)
            self.Re = (I - (K @ C_T)) @ Re_pred

            # Store results
            x_out[k, :] = self.x[:, 0]
            e_out[k, :] = e_k[:, 0]
            self.Re_history.append(self.Re.copy())

            # Keep AdaptiveFilter "w" aligned with current state estimate
            self.w = self.x[:, 0].copy()

        if verbose:
            print(f"Kalman Filtering completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": x_out,
            "errors": e_out,
            "coefficients": self.Re_history,
        }

# EOF
