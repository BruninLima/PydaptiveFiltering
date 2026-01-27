# LatticeRLS.LRLS_priori.py
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
#       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict
from pydaptivefiltering.main import AdaptiveFilter

class LatticeRLS_Priori(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) algorithm using a priori errors.

    This class implements the LRLS algorithm as described in Algorithm 7.4 of the 
    book "Adaptive Filtering: Algorithms and Practical Implementation" by 
    Paulo S. R. Diniz. 

    Attributes:
        filter_order (int): The number of lattice sections (M).
        lam (float): Forgetting factor (0 < lambda <= 1).
        epsilon (float): Small positive constant for energy initialization.
        delta (np.ndarray): Time update of cross-correlation between prediction errors.
        xi_f (np.ndarray): Forward prediction error energy.
        xi_b (np.ndarray): Backward prediction error energy.
        v (np.ndarray): Ladder coefficients for the joint process estimation.
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
        Initializes the LatticeRLS_Priori filter.

        Args:
            filter_order (int): Order of the filter (number of sections).
            lambda_factor (float): Forgetting factor (Lambda). Defaults to 0.99.
            epsilon (float): Regularization factor to avoid singularity. Defaults to 0.1.
            w_init (Optional[Union[np.ndarray, list]]): Initial ladder coefficients (v).
        """
        super().__init__(filter_order, w_init)
        self.lam = lambda_factor
        self.epsilon = epsilon
        self.n_sections = filter_order
        
        self.delta = np.zeros(self.n_sections, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)
        
        if w_init is not None:
            self.v = np.asarray(w_init, dtype=complex)
        else:
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
            Executes the weight update process for the Lattice RLS algorithm (A Priori).

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output of each iteration.
                errors       : Store the error for each iteration.
        """
        tic = time()
        x_in = np.asarray(input_signal, dtype=complex)
        d_in = np.asarray(desired_signal, dtype=complex)
        self._validate_inputs(x_in, d_in)
        
        n_samples = d_in.size
        y = np.zeros(n_samples, dtype=complex)
        e = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            alpha_f = x_in[k]
            alpha_b_curr = np.zeros(self.n_sections + 1, dtype=complex)
            alpha_b_curr[0] = x_in[k]
            
            gamma = 1.0  
            gamma_orders = np.ones(self.n_sections + 1)

            for m in range(self.n_sections):
                gamma_orders[m] = gamma
                
                self.delta[m] = self.lam * self.delta[m] + \
                                (self.error_b_prev[m] * np.conj(alpha_f)) / gamma
                
                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + 1e-12)
                kappa_b = self.delta[m] / (self.xi_f[m] + 1e-12)
                
                alpha_f_next = alpha_f - kappa_f * self.error_b_prev[m]
                alpha_b_curr[m+1] = self.error_b_prev[m] - kappa_b * alpha_f
                
                self.xi_f[m] = self.lam * self.xi_f[m] + \
                               (np.real(alpha_f * np.conj(alpha_f))) / gamma
                self.xi_b[m] = self.lam * self.xi_b[m] + \
                               (np.real(alpha_b_curr[m] * np.conj(alpha_b_curr[m]))) / gamma
                
                gamma = gamma - (np.real(alpha_b_curr[m] * np.conj(alpha_b_curr[m])) / (self.xi_b[m] + 1e-12))
                alpha_f = alpha_f_next

            gamma_orders[self.n_sections] = gamma
            self.xi_f[self.n_sections] = self.lam * self.xi_f[self.n_sections] + \
                                         (np.real(alpha_f * np.conj(alpha_f))) / gamma
            self.xi_b[self.n_sections] = self.lam * self.xi_b[self.n_sections] + \
                                         (np.real(alpha_b_curr[self.n_sections] * np.conj(alpha_b_curr[self.n_sections]))) / gamma

            alpha_e = d_in[k]
            
            for m in range(self.n_sections + 1):
                self.delta_v[m] = self.lam * self.delta_v[m] + \
                                  (alpha_b_curr[m] * np.conj(alpha_e)) / gamma_orders[m]
                
                self.v[m] = self.delta_v[m] / (self.xi_b[m] + 1e-12)
                
                alpha_e = alpha_e - np.conj(self.v[m]) * alpha_b_curr[m]
            
            e_k = alpha_e * gamma
            e[k] = e_k
            y[k] = d_in[k] - e_k

            self.error_b_prev = alpha_b_curr
            
        if verbose:
            print(f"[LRLS-Priori] Completed in {(time() - tic)*1000:.02f} ms")
            
        return {'outputs': y, 'errors': e}