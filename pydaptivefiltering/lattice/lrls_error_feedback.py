# lattice.lrls_error_feedback.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors
#      with Error Feedback.
#      (Algorithm 7.5 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@wam@gmail.com
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict, List
from pydaptivefiltering.base import AdaptiveFilter


class LRLSErrorFeedback(AdaptiveFilter):
    """
    Lattice Recursive Least Squares algorithm with Error Feedback (LRLS-EF).

    Implements Algorithm 7.5 from:
        P. S. R. Diniz, "Adaptive Filtering: Algorithms and Practical Implementation".

    The LRLS-EF structure aims to improve numerical behavior by using an error-feedback
    organization that reduces sensitivity to rounding-error accumulation.

    Implementation / library conventions
    ------------------------------------
    - Complex-valued implementation (supports_complex = True).
    - The algorithm uses a lattice prediction stage and an EF ladder update stage.
    - The ladder coefficient vector is stored in `self.v` (length M+1).
    - For compatibility with AdaptiveFilter:
        * self.w mirrors self.v
        * self._record_history() is called each iteration
        * optimize returns 'coefficients' = self.w_history

    Attributes
    ----------
    lam : float
        Forgetting factor λ.
    epsilon : float
        Regularization/initialization constant.
    n_sections : int
        Filter order M (number of sections); ladder has M+1 taps.

    delta : np.ndarray
        Cross-correlation term used in lattice stage (length M+1).
    xi_f, xi_b : np.ndarray
        Energies used for normalization (length M+2).
    error_b_prev : np.ndarray
        Previous backward error vector (length M+2).

    v : np.ndarray
        Ladder coefficients / joint-process weights (length M+1).
    delta_v : np.ndarray
        Cross-correlation for ladder update (length M+1).
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
        Initializes the LatticeRLSErrorFeedback filter.

        Parameters
        ----------
        filter_order : int
            Order of the filter (number of sections M). Ladder taps are M+1.
        lambda_factor : float
            Forgetting factor λ. Defaults to 0.99.
        epsilon : float
            Regularization factor. Defaults to 0.1.
        w_init : Optional[np.ndarray | list]
            Initial ladder coefficients (length M+1). If None, zeros.
        """
        super().__init__(filter_order, w_init)
        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)

        self.delta = np.zeros(self.n_sections + 1, dtype=complex)  # m = 0..M
        self.xi_f = np.ones(self.n_sections + 2, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 2, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 2, dtype=complex)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

        self._tiny = 1e-12

        self.w = self.v.copy()
        self.w_history = []
        self._record_history()

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the LRLS with Error Feedback (LRLS-EF).
            (Algorithm 7.5 - Diniz)

        Inputs
        -------
            input_signal   : np.ndarray | list
                Input signal x[k].
            desired_signal : np.ndarray | list
                Desired signal d[k].
            verbose        : bool
                Verbose boolean (prints runtime if True).

        Outputs
        -------
            dictionary:
                outputs      : np.ndarray
                    Estimated output y[k].
                errors       : np.ndarray
                    Error signal e[k] = d[k] - y[k].
                coefficients : List[np.ndarray]
                    History of coefficient vectors (ladder taps), aligned with AdaptiveFilter.

        Notes
        -----
        - This implementation is kept close to the original structure you provided,
          but with:
            * safe denominators (epsilon/tiny)
            * gamma clamps (avoid negative or zero)
            * coefficient history recording and standardized return dict.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br
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
            curr_error_b = np.zeros(self.n_sections + 2, dtype=complex)
            curr_error_b[0] = x_in[k]

            self.xi_f[0] = self.lam * self.xi_f[0] + np.real(x_in[k] * np.conj(x_in[k]))
            self.xi_b[0] = self.xi_f[0]

            g_curr = 1.0  

            for m in range(self.n_sections + 1):
                denom_g = max(g_curr, self._tiny)

                self.delta[m] = (
                    self.lam * self.delta[m]
                    + (self.error_b_prev[m] * np.conj(err_f)) / denom_g
                )

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + self._tiny)
                kappa_b = self.delta[m] / (self.xi_f[m] + self._tiny)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_error_b[m + 1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m + 1] = (
                    self.lam * self.xi_f[m + 1]
                    + np.real(new_err_f * np.conj(new_err_f)) / denom_g
                )
                self.xi_b[m + 1] = (
                    self.lam * self.xi_b[m + 1]
                    + np.real(curr_error_b[m + 1] * np.conj(curr_error_b[m + 1])) / denom_g
                )

                g_next = g_curr - (
                    np.real(self.error_b_prev[m] * np.conj(self.error_b_prev[m]))
                    / (self.xi_b[m] + self._tiny)
                )
                g_curr = max(g_next, self._tiny)
                err_f = new_err_f

            y_k = np.vdot(self.v, curr_error_b[: self.n_sections + 1])
            y[k] = y_k
            e_k = d_in[k] - y_k
            e[k] = e_k

            g_ladder = 1.0
            for m in range(self.n_sections + 1):
                denom_gl = max(g_ladder, self._tiny)

                self.delta_v[m] = (
                    self.lam * self.delta_v[m]
                    + (curr_error_b[m] * np.conj(d_in[k])) / denom_gl
                )

                self.v[m] = self.delta_v[m] / (self.xi_b[m] + self._tiny)

                g_next_ladder = g_ladder - (
                    np.real(curr_error_b[m] * np.conj(curr_error_b[m]))
                    / (self.xi_b[m] + self._tiny)
                )
                g_ladder = max(g_next_ladder, self._tiny)

            self.error_b_prev = curr_error_b


            self.w = self.v.copy()
            self._record_history()

        if verbose:
            print(f"[LRLS-EF] Completed in {(time() - tic) * 1000:.02f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
# EOF