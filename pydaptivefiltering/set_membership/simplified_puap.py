#  set_membership.simplified_puap.py
#
#       Implements the Simplified Set-membership Partial-Update
#       Affine-Projection (SM-Simp-PUAP) algorithm for COMPLEX valued data.
#       (Algorithm 6.6 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import perf_counter
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class SimplifiedSMPUAP(AdaptiveFilter):
    """
    Simplified Set-membership Partial-Update Affine-Projection (SM-Simp-PUAP) adaptive filter
    (complex-valued).

    Set-membership affine-projection adaptive filter with *partial updates*, following
    Diniz (Alg. 6.6). At each iteration, the algorithm forms an affine-projection (AP)
    a priori error vector from a sliding regressor matrix. An update is performed only
    when the magnitude of the first a priori error component exceeds a prescribed bound
    (set-membership condition). When an update occurs, only a subset of coefficients is
    updated according to a user-provided selector mask.

    Parameters
    ----------
    filter_order : int
        FIR filter order (number of taps minus 1). The number of coefficients is
        ``M+1 = filter_order + 1``.
    gamma_bar : float
        Error magnitude threshold for triggering an update. An update is performed when
        ``|e[k]| > gamma_bar`` where ``e[k]`` is the first AP a priori error component.
    gamma : float
        Regularization factor added to the AP correlation matrix (diagonal loading).
        Must typically be positive to improve numerical stability.
    L : int
        Projection order / reuse data factor. The AP vectors have length ``L+1`` and
        the regressor matrix has shape ``(M+1, L+1)``.
    up_selector : array_like of {0,1}
        Partial-update selector matrix with shape ``(M+1, N)``. Column ``k`` (a vector
        of length ``M+1``) selects which coefficients are updated at iteration ``k``.
        Non-selected coefficients remain unchanged in that iteration.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M+1,)``. If None, initializes
        with zeros (via the base class).

    Notes
    -----
    Complex-valued
        This implementation supports complex-valued signals and coefficients
        (``supports_complex=True``).

    Regressor matrix and AP vectors (as implemented)
        Let ``M+1`` be the number of coefficients and ``L+1`` the projection length.
        The regressor matrix ``X[k]`` is built by stacking the most recent tapped-delay
        input vectors:

        .. math::
            X[k] = [x_k, x_{k-1}, \\dots, x_{k-L}] \\in \\mathbb{C}^{(M+1)\\times(L+1)},

        where each column is an ``(M+1)``-length FIR regressor built from the input signal.
        The AP a priori output vector and error vector (conjugated form) are computed as:

        .. math::
            y^*[k] = X^H[k] w[k-1], \\qquad
            e^*[k] = d^*[k] - y^*[k],

        producing vectors in :math:`\\mathbb{C}^{(L+1)}`. This implementation returns only
        the *first component* as the scalar output/error:

        .. math::
            y[k] = y^*[k]_0, \\qquad e[k] = e^*[k]_0.

    Set-membership update gate (as implemented)
        The update step-size ``mu[k]`` is defined by:

        .. math::
            \\mu[k] =
            \\begin{cases}
                1 - \\frac{\\bar\\gamma}{|e[k]|}, & |e[k]| > \\bar\\gamma \\\\
                0, & \\text{otherwise}
            \\end{cases}

        where ``bar_gamma = gamma_bar``.

    Partial-update mechanism (as implemented)
        Let ``c[k]`` be the selector column (shape ``(M+1,1)``). The selected regressor
        matrix is formed by element-wise selection:

        .. math::
            C_X[k] = \\operatorname{diag}(c[k])\\,X[k],

        implemented as ``C_reg = c_vec * regressor_matrix``. The regularized correlation
        matrix is

        .. math::
            R[k] = X^H[k] C_X[k] + \\gamma I,

        and the coefficient update uses the AP unit vector ``u_1 = [1, 0, ..., 0]^T`` to
        target the first error component:

        .. math::
            w[k] = w[k-1] + C_X[k] R^{-1}[k] (\\mu[k] e[k] u_1).

    Implementation details
        - ``up_selector`` must provide at least ``N`` columns for an ``N``-sample run.
        - ``np.linalg.solve`` is used for the linear system; if singular/ill-conditioned,
          a pseudoinverse fallback is used.
        - Coefficient history is recorded by the base class at every iteration.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, 5th ed., Algorithm 6.6.
    """
    supports_complex: bool = True

    gamma_bar: float
    gamma: float
    L: int
    n_coeffs: int

    def __init__(
        self,
        filter_order: int,
        gamma_bar: float,
        gamma: float,
        L: int,
        up_selector: Union[np.ndarray, list],
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            FIR filter order (number of taps - 1). Number of coefficients is filter_order + 1.
        gamma_bar:
            Error magnitude threshold for triggering updates.
        gamma:
            Regularization factor for the AP correlation matrix.
        L:
            Reuse data factor / constraint length (projection order).
        up_selector:
            Partial-update selector matrix with shape (M+1, N), entries in {0,1}.
            Each column selects which coefficients are updated at iteration k.
        w_init:
            Optional initial coefficient vector. If None, initializes to zeros.
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.gamma_bar = float(gamma_bar)
        self.gamma = float(gamma)
        self.L = int(L)
        self.n_coeffs = int(self.filter_order + 1)

        sel = np.asarray(up_selector)
        if sel.ndim != 2:
            raise ValueError("up_selector must be a 2D array with shape (M+1, N).")
        if sel.shape[0] != self.n_coeffs:
            raise ValueError(
                f"up_selector must have shape (M+1, N) with M+1={self.n_coeffs}, got {sel.shape}."
            )
        self.up_selector: np.ndarray = sel

        self.regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, self.L + 1), dtype=complex)
        
        self.n_updates: int = 0

    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the SM-Simp-PUAP adaptation loop over paired input/desired sequences.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : array_like of complex
            Desired sequence ``d[k]`` with shape ``(N,)`` (will be flattened).
        verbose : bool, optional
            If True, prints the total runtime and the number of performed updates.
        return_internal_states : bool, optional
            If True, includes internal trajectories in ``result.extra``:
            ``"mu"`` and ``"selected_count"`` in addition to the always-present
            set-membership bookkeeping fields.

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar a priori output sequence (first component of the AP output vector).
            - errors : ndarray of complex, shape ``(N,)``
                Scalar a priori error sequence (first component of the AP error vector).
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"a_priori"``.
            - extra : dict
                Always includes ``"n_updates"`` and ``"update_mask"``. If
                ``return_internal_states=True``, also includes ``"mu"`` and
                ``"selected_count"``.
        """
        tic: float = perf_counter()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        d: np.ndarray = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples: int = int(d.size)
        n_coeffs: int = int(self.n_coeffs)
        Lp1: int = int(self.L + 1)

        if self.up_selector.shape[1] < n_samples:
            raise ValueError(
                f"up_selector has {self.up_selector.shape[1]} columns, but signal has {n_samples} samples."
            )

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        update_mask: np.ndarray = np.zeros(n_samples, dtype=bool)

        mu_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=float) if return_internal_states else None
        selcnt_track: Optional[np.ndarray] = np.zeros(n_samples, dtype=int) if return_internal_states else None

        self.n_updates = 0
        w_current: np.ndarray = self.w.astype(complex, copy=False).reshape(-1, 1)

        prefixed_input: np.ndarray = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x])
        prefixed_desired: np.ndarray = np.concatenate([np.zeros(self.L, dtype=complex), d])

        u1: np.ndarray = np.zeros((Lp1, 1), dtype=complex)
        u1[0, 0] = 1.0

        for k in range(n_samples):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]
            start_idx = k + n_coeffs - 1
            stop = (k - 1) if (k > 0) else None
            self.regressor_matrix[:, 0] = prefixed_input[start_idx:stop:-1]

            output_ap_conj: np.ndarray = (self.regressor_matrix.conj().T) @ w_current
            desired_slice = prefixed_desired[k + self.L : stop : -1]
            error_ap_conj: np.ndarray = desired_slice.conj().reshape(-1, 1) - output_ap_conj

            yk: complex = complex(output_ap_conj[0, 0])
            ek: complex = complex(error_ap_conj[0, 0])

            outputs[k] = yk
            errors[k] = ek

            eabs: float = float(np.abs(ek))
            if eabs > self.gamma_bar:
                self.n_updates += 1
                update_mask[k] = True
                mu: float = float(1.0 - (self.gamma_bar / eabs))
            else:
                mu = 0.0

            c_vec: np.ndarray = self.up_selector[:, k].reshape(-1, 1).astype(float)

            if return_internal_states and selcnt_track is not None:
                selcnt_track[k] = int(np.sum(c_vec != 0))

            if mu > 0.0:
                C_reg: np.ndarray = c_vec * self.regressor_matrix  # (M+1, L+1)

                R: np.ndarray = (self.regressor_matrix.conj().T @ C_reg) + self.gamma * np.eye(Lp1)

                rhs: np.ndarray = mu * ek * u1  # (L+1,1)

                try:
                    inv_term = np.linalg.solve(R, rhs)
                except np.linalg.LinAlgError:
                    inv_term = np.linalg.pinv(R) @ rhs

                w_current = w_current + (C_reg @ inv_term)

            if return_internal_states and mu_track is not None:
                mu_track[k] = mu

            self.w = w_current.ravel()
            self._record_history()

        runtime_s: float = perf_counter() - tic
        if verbose:
            print(f"[SM-Simp-PUAP] Updates: {self.n_updates}/{n_samples} | Runtime: {runtime_s * 1000:.2f} ms")

        extra: Dict[str, Any] = {
            "n_updates": int(self.n_updates),
            "update_mask": update_mask,
        }
        if return_internal_states:
            extra.update(
                {
                    "mu": mu_track,
                    "selected_count": selcnt_track,
                }
            )

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_priori",
            extra=extra,
        )
# EOF