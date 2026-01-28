# blind.affine_projection_cm.py
#
#       Implements the Complex Affine-Projection Constant-Modulus algorithm
#       for COMPLEX valued data.
#       (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermeodeoliveirapinto@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult


class AffineProjectionCM(AdaptiveFilter):
    """
    Complex Affine-Projection Constant-Modulus (AP-CM) adaptive filter.

    Blind affine-projection algorithm based on the constant-modulus criterion,
    following Diniz (Alg. 13.4). This implementation uses a *unit-modulus*
    reference (i.e., target magnitude equal to 1) obtained by normalizing the
    affine-projection output vector.

    Parameters
    ----------
    filter_order : int, optional
        Adaptive FIR filter order ``M``. The number of coefficients is ``M + 1``.
        Default is 5.
    step_size : float, optional
        Adaptation step size ``mu``. Default is 0.1.
    memory_length : int, optional
        Reuse factor ``L`` (number of past regressors reused). The affine-
        projection block size is therefore ``P = L + 1`` columns. Default is 2.
    gamma : float, optional
        Levenberg-Marquardt regularization factor ``gamma`` used in the
        ``(L + 1) x (L + 1)`` normal-equation system for numerical stability.
        Default is 1e-6.
    w_init : array_like of complex, optional
        Initial coefficient vector ``w(0)`` with shape ``(M + 1,)``. If None,
        initializes with zeros.

    Notes
    -----
    At iteration ``k``, form the regressor block matrix:

    - ``X(k) ∈ C^{(M+1) x (L+1)}``, whose columns are the most recent regressor
    vectors (newest in column 0).

    The affine-projection output vector is:

    .. math::
        y_{ap}(k) = X^H(k) w(k)  \\in \\mathbb{C}^{L+1}.

    This implementation uses a *unit-circle projection* (normalization) as the
    constant-modulus "reference":

    .. math::
        d_{ap}(k) = \\frac{y_{ap}(k)}{|y_{ap}(k)|},

    applied element-wise, with a small threshold to avoid division by zero.

    The error vector is:

    .. math::
        e_{ap}(k) = d_{ap}(k) - y_{ap}(k).

    The update direction ``g(k)`` is obtained by solving the regularized system:

    .. math::
        (X^H(k) X(k) + \\gamma I_{L+1})\\, g(k) = e_{ap}(k),

    and the coefficient update is:

    .. math::
        w(k+1) = w(k) + \\mu X(k) g(k).

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
    Implementation*, 5th ed., Algorithm 13.4.
    """
    supports_complex: bool = True
    step_size: float
    memory_length: int
    gamma: float
    n_coeffs: int

    def __init__(
        self,
        filter_order: int = 5,
        step_size: float = 0.1,
        memory_length: int = 2,
        gamma: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init=w_init)
        self.step_size = float(step_size)
        self.memory_length = int(memory_length)
        self.gamma = float(gamma)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Optional[Union[np.ndarray, list]] = None,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the AP-CM adaptation loop over an input sequence.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)`` (will be flattened).
        desired_signal : None, optional
            Ignored. This is a blind algorithm: the reference is derived from
            the output via unit-modulus normalization.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, includes the last internal states in ``result.extra``:
            ``"last_update_factor"`` (``g(k)``) and ``"last_regressor_matrix"``
            (``X(k)``).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Scalar output sequence, ``y[k] = y_ap(k)[0]``.
            - errors : ndarray of complex, shape ``(N,)``
                Scalar CM error sequence, ``e[k] = e_ap(k)[0]``.
            - coefficients : ndarray of complex
                Coefficient history recorded by the base class.
            - error_type : str
                Set to ``"blind_constant_modulus"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True``.
        """
        tic: float = time()

        x: np.ndarray = np.asarray(input_signal, dtype=complex).ravel()
        n_samples: int = int(x.size)

        outputs: np.ndarray = np.zeros(n_samples, dtype=complex)
        errors: np.ndarray = np.zeros(n_samples, dtype=complex)

        L: int = int(self.memory_length)

        regressor_matrix: np.ndarray = np.zeros((self.n_coeffs, L + 1), dtype=complex)
        I_reg: np.ndarray = (self.gamma * np.eye(L + 1)).astype(complex)

        x_padded: np.ndarray = np.zeros(n_samples + self.filter_order, dtype=complex)
        x_padded[self.filter_order:] = x

        last_update_factor: Optional[np.ndarray] = None

        for k in range(n_samples):
            regressor_matrix[:, 1:] = regressor_matrix[:, :-1]
            regressor_matrix[:, 0] = x_padded[k : k + self.filter_order + 1][::-1]

            output_ap: np.ndarray = np.dot(np.conj(regressor_matrix).T, self.w)

            abs_out: np.ndarray = np.abs(output_ap)
            desired_level: np.ndarray = np.zeros_like(output_ap, dtype=complex)
            np.divide(output_ap, abs_out, out=desired_level, where=abs_out > 1e-12)

            error_ap: np.ndarray = desired_level - output_ap

            phi: np.ndarray = np.dot(np.conj(regressor_matrix).T, regressor_matrix) + I_reg
            update_factor: np.ndarray = np.linalg.solve(phi, error_ap)
            last_update_factor = update_factor

            self.w = self.w + self.step_size * np.dot(regressor_matrix, update_factor)

            outputs[k] = output_ap[0]
            errors[k] = error_ap[0]

            self._record_history()

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[AffineProjectionCM] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "last_update_factor": last_update_factor,
                "last_regressor_matrix": regressor_matrix.copy(),
            }

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="blind_constant_modulus",
            extra=extra,
        )
# EOF