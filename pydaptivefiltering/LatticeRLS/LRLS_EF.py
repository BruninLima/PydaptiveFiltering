# LatticeRLS.LRLS_EF.py
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
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br

import numpy as np
from time import time
from typing import Optional, Union, Dict
from pydaptivefiltering.main import AdaptiveFilter

class LatticeRLSErrorFeedback(AdaptiveFilter):
    """
    Lattice Recursive Least Squares algorithm with Error Feedback (LRLS-EF).

    This class implements the LRLS-EF algorithm as described in Algorithm 7.5 of 
    "Adaptive Filtering: Algorithms and Practical Implementation" (Diniz). 
    The Error Feedback structure is designed to improve numerical stability by 
    directly updating the reflection and ladder coefficients using the error 
    signals, making it less susceptible to accumulation of rounding errors 
    compared to the standard a posteriori version.

    Attributes:
        filter_order (int): The number of lattice sections (M).
        lam (float): Forgetting factor (0 < lambda <= 1).
        epsilon (float): Small positive constant for energy initialization.
        xi_f (np.ndarray): Forward prediction error energy.
        xi_b (np.ndarray): Backward prediction error energy.
        gamma (np.ndarray): Conversion factor (likelihood variable).
        v (np.ndarray): Ladder coefficients (joint process weights).
        delta (np.ndarray): Cross-correlation between prediction errors.
        delta_v (np.ndarray): Cross-correlation between backward errors and desired signal.
    """

    def __init__(
        self, 
        filter_order: int, 
        lambda_factor: float = 0.99, 
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Initializes the LatticeRLSErrorFeedback filter.

        Args:
            filter_order (int): Order of the filter (number of sections).
            lambda_factor (float): Forgetting factor (Lambda). Defaults to 0.99.
            epsilon (float): Regularization factor. Defaults to 0.1.
            w_init (Optional[Union[np.ndarray, list]]): Initial ladder coefficients.
        """
        super().__init__(filter_order, w_init)
        self.lam = lambda_factor
        self.epsilon = epsilon
        self.n_sections = filter_order
        
        self.delta = np.zeros(self.n_sections + 1, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 2, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 2, dtype=float) * self.epsilon
        self.gamma = np.ones(self.n_sections + 2, dtype=float)
        self.error_b_prev = np.zeros(self.n_sections + 2, dtype=complex)
        
        self.v = np.zeros(self.n_sections + 1, dtype=complex)
        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Description
        -----------
            Executes the weight update process for the LRLS with Error Feedback.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input signal x)
            desired_signal : np.ndarray | list (Desired signal d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output.
                errors       : Store the final error.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                                diniz@lps.ufrj.br
        """
        tic = time()
        x_in = np.asarray(input_signal, dtype=complex)
        d_in = np.asarray(desired_signal, dtype=complex)
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
                self.delta[m] = self.lam * self.delta[m] + \
                                (self.error_b_prev[m] * np.conj(err_f)) / g_curr

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + 1e-12)
                kappa_b = self.delta[m] / (self.xi_f[m] + 1e-12)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_error_b[m+1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m+1] = self.lam * self.xi_f[m+1] + \
                                 np.real(new_err_f * np.conj(new_err_f)) / g_curr
                self.xi_b[m+1] = self.lam * self.xi_b[m+1] + \
                                 np.real(curr_error_b[m+1] * np.conj(curr_error_b[m+1])) / g_curr

                g_next = g_curr - (np.real(self.error_b_prev[m] * np.conj(self.error_b_prev[m])) / (self.xi_b[m] + 1e-12))
                
                err_f = new_err_f
                g_curr = g_next

            y_k = 0j
            for m in range(self.n_sections + 1):
                y_k += np.conj(self.v[m]) * curr_error_b[m]
            
            y[k] = y_k
            e_k = d_in[k] - y_k
            e[k] = e_k
            
            g_ladder = 1.0
            for m in range(self.n_sections + 1):
                self.delta_v[m] = self.lam * self.delta_v[m] + \
                                  (curr_error_b[m] * np.conj(d_in[k])) / g_ladder
                
                self.v[m] = self.delta_v[m] / (self.xi_b[m] + 1e-12)
                
                g_next_ladder = g_ladder - (np.real(curr_error_b[m] * np.conj(curr_error_b[m])) / (self.xi_b[m] + 1e-12))
                g_ladder = g_next_ladder

            self.error_b_prev = curr_error_b
            
        if verbose:
            print(f"[LRLS-EF] Completed in {(time() - tic)*1000:.02f} ms")
            
        return {'outputs': y, 'errors': e}

# EOF