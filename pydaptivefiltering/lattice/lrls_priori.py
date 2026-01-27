# lattice.lrls_priori.py
#
#      Implements the Lattice RLS algorithm based on a priori errors.
#      (Algorithm 7.4 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict, List
from pydaptivefiltering.base import AdaptiveFilter


class LRLSPriori(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) algorithm using a priori errors.

    This implementation follows Algorithm 7.4 from:
        P. S. R. Diniz, "Adaptive Filtering: Algorithms and Practical Implementation".

    The lattice stage performs orthogonal prediction of the input signal using
    forward and backward errors, while the ladder stage estimates the desired
    signal using a priori errors.

    Notes
    -----
    - Complex-valued implementation (supports_complex = True).
    - Ladder coefficients are stored in `self.v` (length M+1).
    - For compatibility with the AdaptiveFilter base class, `self.w` mirrors `self.v`.
    - The full coefficient history is stored in `self.w_history`.

    Attributes
    ----------
    filter_order : int
        Number of lattice sections M (number of ladder taps is M+1).
    lam : float
        Forgetting factor (0 < λ ≤ 1).
    epsilon : float
        Regularization constant for energy initialization.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        super().__init__(filter_order, w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)

        self.delta = np.zeros(self.n_sections, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)

        if w_init is not None:
            self.v = np.asarray(w_init, dtype=complex).reshape(-1)
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.w = self.v.copy()
        self.w_history = []
        self._record_history()

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

        self._tiny = 1e-12

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Executes the weight update process for the Lattice RLS algorithm
        using a priori errors (Algorithm 7.4).

        Parameters
        ----------
        input_signal : np.ndarray | list
            Input signal x[k].
        desired_signal : np.ndarray | list
            Desired signal d[k].
        verbose : bool
            If True, prints execution time.

        Returns
        -------
        dict
            Dictionary containing:
                - outputs : np.ndarray
                    Estimated output signal y[k].
                - errors : np.ndarray
                    A priori error signal e[k].
                - coefficients : List[np.ndarray]
                    History of ladder coefficient vectors.

        Authors
        -------
            . Bruno Ramos Lima Netto
            . Guilherme de Oliveira Pinto
            . Markus Vinícius Santos Lima
            . Wallace Alves Martins
            . Luiz Wagner Pereira Biscainho
            . Paulo Sergio Ramirez Diniz
        """
        tic = time()

        x_in = np.asarray(input_signal, dtype=complex).reshape(-1)
        d_in = np.asarray(desired_signal, dtype=complex).reshape(-1)
        self._validate_inputs(x_in, d_in)

        n_samples = d_in.size
        y = np.zeros(n_samples, dtype=complex)
        e = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            alpha_f = x_in[k]
            alpha_b = np.zeros(self.n_sections + 1, dtype=complex)
            alpha_b[0] = x_in[k]

            gamma = 1.0
            gamma_orders = np.ones(self.n_sections + 1)

            for m in range(self.n_sections):
                gamma_orders[m] = gamma

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(alpha_f)) / max(gamma, self._tiny)
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + self._tiny)
                kappa_b = self.delta[m] / (self.xi_f[m] + self._tiny)

                alpha_f_next = alpha_f - kappa_f * self.error_b_prev[m]
                alpha_b[m + 1] = self.error_b_prev[m] - kappa_b * alpha_f

                self.xi_f[m] = (
                    self.lam * self.xi_f[m]
                    + np.real(alpha_f * np.conj(alpha_f)) / max(gamma, self._tiny)
                )
                self.xi_b[m] = (
                    self.lam * self.xi_b[m]
                    + np.real(alpha_b[m] * np.conj(alpha_b[m])) / max(gamma, self._tiny)
                )

                gamma_next = gamma - (
                    np.real(alpha_b[m] * np.conj(alpha_b[m])) / (self.xi_b[m] + self._tiny)
                )
                gamma = max(gamma_next, self._tiny)
                alpha_f = alpha_f_next

            gamma_orders[self.n_sections] = gamma

            self.xi_f[self.n_sections] = (
                self.lam * self.xi_f[self.n_sections]
                + np.real(alpha_f * np.conj(alpha_f)) / max(gamma, self._tiny)
            )
            self.xi_b[self.n_sections] = (
                self.lam * self.xi_b[self.n_sections]
                + np.real(alpha_b[self.n_sections] * np.conj(alpha_b[self.n_sections]))
                / max(gamma, self._tiny)
            )

            alpha_e = d_in[k]

            for m in range(self.n_sections + 1):
                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (alpha_b[m] * np.conj(alpha_e)) / max(gamma_orders[m], self._tiny)
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)
                alpha_e = alpha_e - np.conj(self.v[m]) * alpha_b[m]

            e_k = alpha_e * gamma
            e[k] = e_k
            y[k] = d_in[k] - e_k

            self.error_b_prev = alpha_b

            self.w = self.v.copy()
            self._record_history()

        if verbose:
            print(f"[LRLS-Priori] Completed in {(time() - tic) * 1000:.02f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
#EOF