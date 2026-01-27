# LatticeRLS.NLRLS_pos.py
#
#      Implements the Normalized Lattice RLS algorithm based on a posteriori error.
#      (Algorithm 7.6 - book: Adaptive Filtering: Algorithms and Practical
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
from typing import Optional, Union, Dict
from pydaptivefiltering.main import AdaptiveFilter

class NormalizedLatticeRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Normalized Lattice RLS algorithm based on a posteriori error.
        (Algorithm 7.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
        
        This algorithm is characterized by its superior numerical properties, as all 
        internal variables (normalized errors and reflection coefficients) are 
        magnitude-bounded by unity.
    """

    def __init__(
        self, 
        filter_order: int, 
        lambda_factor: float = 0.99, 
        epsilon: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order  : int (The order of the filter M)
            lambda_factor : float (Forgetting factor Lambda)
            epsilon       : float (Regularization factor for initialization)
            w_init        : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.lam = lambda_factor
        self.epsilon = epsilon
        self.n_sections = filter_order
        
        self.rho = np.zeros(self.n_sections, dtype=complex)       
        self.rho_v = np.zeros(self.n_sections + 1, dtype=complex)  
        self.bar_b_prev = np.zeros(self.n_sections + 1, dtype=complex) 
        self.xi_half = np.sqrt(epsilon) 

    def _safe_sqrt(self, value: float) -> float:
        """
        Ensures the value for sqrt is not negative due to precision errors.
        """
        return np.sqrt(max(0.0, float(value)))

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Description
        -----------
            Executes the weight update process for the Normalized Lattice RLS.

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
        x_in = np.asarray(input_signal, dtype=complex)
        d_in = np.asarray(desired_signal, dtype=complex)
        self._validate_inputs(x_in, d_in)
        
        n_samples = d_in.size
        y = np.zeros(n_samples, dtype=complex)
        e = np.zeros(n_samples, dtype=complex)

        sqrt_lam = np.sqrt(self.lam)

        for k in range(n_samples):
            self.xi_half = np.sqrt(self.lam * (self.xi_half**2) + np.abs(x_in[k])**2 + self.epsilon)
            
            bar_f = x_in[k] / (self.xi_half + self.epsilon)
            if np.abs(bar_f) > 1.0: 
                bar_f /= (np.abs(bar_f) + self.epsilon)
            
            bar_b_curr = np.zeros(self.n_sections + 1, dtype=complex)
            bar_b_curr[0] = bar_f
            
            for m in range(self.n_sections):
                cos_f = self._safe_sqrt(1 - np.abs(bar_f)**2)
                cos_b = self._safe_sqrt(1 - np.abs(self.bar_b_prev[m])**2)

                self.rho[m] = sqrt_lam * cos_f * cos_b * self.rho[m] + np.conj(bar_f) * self.bar_b_prev[m]

                if np.abs(self.rho[m]) >= 1.0:
                    self.rho[m] /= (np.abs(self.rho[m]) + self.epsilon)
                
                cos_rho = self._safe_sqrt(1 - np.abs(self.rho[m])**2)
                
                denom_f = (cos_rho * cos_b) + self.epsilon
                denom_b = (cos_rho * cos_f) + self.epsilon
                
                f_next = (bar_f - self.rho[m] * self.bar_b_prev[m]) / denom_f
                b_next = (self.bar_b_prev[m] - np.conj(self.rho[m]) * bar_f) / denom_b
                
                bar_f = f_next
                bar_b_curr[m+1] = b_next

            bar_e = d_in[k] / (self.xi_half + self.epsilon)
            if np.abs(bar_e) > 1.0: 
                bar_e /= (np.abs(bar_e) + self.epsilon)

            for m in range(self.n_sections + 1):
                cos_e = self._safe_sqrt(1 - np.abs(bar_e)**2)
                cos_b = self._safe_sqrt(1 - np.abs(bar_b_curr[m])**2)

                self.rho_v[m] = sqrt_lam * cos_e * cos_b * self.rho_v[m] + np.conj(bar_e) * bar_b_curr[m]
                
                if np.abs(self.rho_v[m]) >= 1.0:
                    self.rho_v[m] /= (np.abs(self.rho_v[m]) + self.epsilon)
                
                cos_rho_v = self._safe_sqrt(1 - np.abs(self.rho_v[m])**2)

                bar_e = (bar_e - self.rho_v[m] * bar_b_curr[m]) / ((cos_rho_v * cos_b) + self.epsilon)

            e[k] = bar_e * self.xi_half
            y[k] = d_in[k] - e[k]

            self.bar_b_prev = bar_b_curr.copy()
            
        if verbose:
            print(f"[NLRLS] Completed in {(time() - tic)*1000:.02f} ms")
            
        return {'outputs': y, 'errors': e}

# EOF