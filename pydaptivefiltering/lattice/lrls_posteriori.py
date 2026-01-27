# lattice.lrls_posteriori.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors.
#      (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict, List
from pydaptivefiltering.base import AdaptiveFilter


class LRLSPosteriori(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) algorithm using a posteriori errors.

    Implements Algorithm 7.1 (Diniz) in a lattice structure (prediction + ladder).
    The lattice stage orthogonalizes the input through forward/backward prediction errors.
    The ladder stage estimates the desired signal using backward prediction errors.

    Notes on implementation / conventions:
    - The algorithm is implemented in complex arithmetic (supports_complex=True).
    - The “ladder coefficients” are stored in self.v (length M+1).
    - For compatibility with the pydaptivefiltering base class, self.w mirrors self.v.
      This allows filter_signal() and w_history usage consistently across the library.

    Attributes
    ----------
    m : int
        Filter order (M), thus number of taps is M+1.
    w : np.ndarray
        Current coefficient vector (mirrors v).
    w_history : List[np.ndarray]
        History of coefficient vectors over time.

    Algorithm variables
    -------------------
    lam : float
        Forgetting factor λ (0 < λ <= 1).
    epsilon : float
        Regularization / initialization for energies.
    n_sections : int
        Number of lattice sections (= filter order M).

    delta : np.ndarray
        Cross-correlation update term per section (length M).
    xi_f : np.ndarray
        Forward prediction error energy per stage (length M+1).
    xi_b : np.ndarray
        Backward prediction error energy per stage (length M+1).
    error_b_prev : np.ndarray
        Previous backward errors vector (length M+1).

    v : np.ndarray
        Ladder coefficient vector (length M+1).
    delta_v : np.ndarray
        Cross-correlation between backward error and a posteriori error (length M+1).
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Initializes the LatticeRLS filter.

        Parameters
        ----------
        filter_order : int
            Number of lattice sections M. Number of ladder taps is M+1.
        lambda_factor : float
            Forgetting factor λ. Defaults to 0.99.
        epsilon : float
            Initialization for energies (xi_f and xi_b). Defaults to 0.1.
        w_init : Optional[np.ndarray | list]
            Optional initial ladder coefficients (length M+1). If None, zeros.
        """
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
        Executes the LRLS adaptation.

        Parameters
        ----------
        input_signal : np.ndarray | list
            Input signal x[k] (1D).
        desired_signal : np.ndarray | list
            Desired signal d[k] (1D).
        verbose : bool
            If True, prints runtime.

        Returns
        -------
        dict:
            outputs : np.ndarray
                Filter output y[k].
            errors : np.ndarray
                A posteriori error e[k] = d[k] - y[k] after ladder update.
            coefficients : List[np.ndarray]
                History of coefficient vectors (ladder taps), aligned with AdaptiveFilter.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic = time()

        x_in = np.asarray(input_signal, dtype=complex).reshape(-1)
        d_in = np.asarray(desired_signal, dtype=complex).reshape(-1)
        self._validate_inputs(x_in, d_in)

        n_samples = d_in.size
        y = np.zeros(n_samples, dtype=complex)
        e = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):

            err_f = x_in[k]
            curr_err_b = np.zeros(self.n_sections + 1, dtype=complex)
            curr_err_b[0] = x_in[k]

            self.xi_f[0] = self.lam * self.xi_f[0] + np.real(err_f * np.conj(err_f))
            self.xi_b[0] = self.xi_f[0]

            gamma_m = 1.0 

            for m in range(self.n_sections):
                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(err_f)) / max(gamma_m, self._tiny)
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + self._tiny)
                kappa_b = self.delta[m] / (self.xi_f[m] + self._tiny)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_err_b[m + 1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m + 1] = self.xi_f[m] - np.real(kappa_f * self.delta[m])
                self.xi_b[m + 1] = self.xi_b[m] - np.real(kappa_b * np.conj(self.delta[m]))

                denom = self.xi_b[m] + self._tiny
                gamma_m_next = gamma_m - (np.real(curr_err_b[m] * np.conj(curr_err_b[m])) / denom)

                gamma_m = max(gamma_m_next, self._tiny)
                err_f = new_err_f

            e_posteriori = d_in[k]
            gamma_ladder = 1.0

            for m in range(self.n_sections + 1):
                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (curr_err_b[m] * np.conj(e_posteriori)) / max(gamma_ladder, self._tiny)
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)

                e_posteriori = e_posteriori - np.conj(self.v[m]) * curr_err_b[m]

                denom = self.xi_b[m] + self._tiny
                gamma_ladder_next = gamma_ladder - (np.real(curr_err_b[m] * np.conj(curr_err_b[m])) / denom)
                gamma_ladder = max(gamma_ladder_next, self._tiny)

            y[k] = d_in[k] - e_posteriori
            e[k] = e_posteriori

            self.error_b_prev = curr_err_b

            self.w = self.v.copy()
            self._record_history()

        if verbose:
            print(f"[LatticeRLS] Completed in {(time() - tic) * 1000:.02f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
#EOF