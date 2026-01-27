#  Volterra_RLS.py
#
#       Implements the Volterra RLS algorithm for REAL valued data.
#       (Algorithm 11.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals

class VolterraRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Volterra RLS algorithm for REAL valued data.
        (Algorithm 11.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        memory: int = 3, 
        forgetting_factor: float = 0.98, 
        delta: float = 1.0,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            memory            : int (The linear memory length L. 
                                Total coefficients for 2nd order: L + L*(L+1)/2)
            forgetting_factor : float (The lambda factor, typically between 0.9 and 1)
            delta             : float (Regularization factor to initialize S_d matrix)
            w_init            : array_like, optional (Initial coefficients)
        """
        self.memory: int = memory
        self.forgetting_factor: float = forgetting_factor
        
        # Total coefficients calculation (same logic as Volterra LMS)
        # For L=3, Nw = 9
        n_coeffs = memory + (memory * (memory + 1)) // 2

        super().__init__(m = n_coeffs-1, w_init=w_init)
        
        # Initialize Inverse Correlation Matrix Sd = delta^-1 * I
        self.S_d: np.ndarray = np.eye(n_coeffs) / delta

    def _create_volterra_regressor(self, x_lin: np.ndarray) -> np.ndarray:
        """
        Constructs the second-order Volterra regressor.
        Matches the order: [linear terms, quadratic terms (i <= j)]
        """
        quad_terms = []
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad_terms.append(x_lin[i] * x_lin[j])
        
        return np.concatenate([x_lin, np.array(quad_terms)])

    @ensure_real_signals
    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Volterra RLS algorithm.

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
                coefficients : Store the estimated coefficients for each iteration.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal, dtype=float)
        d: np.ndarray = np.asarray(desired_signal, dtype=float)

        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=float)
        e: np.ndarray = np.zeros(n_samples, dtype=float)
        
        # Padding for linear delay line
        x_padded: np.ndarray = np.zeros(n_samples + self.memory - 1, dtype=float)
        x_padded[self.memory - 1:] = x

        for k in range(n_samples):
            # 1. Linear delay line extraction
            x_lin = x_padded[k : k + self.memory][::-1]
            
            # 2. Volterra regressor expansion (uxl)
            uxl = self._create_volterra_regressor(x_lin)
            
            # 3. Prior Error (elinha)
            # elinha(i) = d(i) - w(:,i)' * uxl(:,i)
            e_prior = d[k] - np.dot(self.w, uxl)
            
            # 4. Update Inverse Correlation Matrix (Sd)
            psi = np.dot(self.S_d, uxl)
            den = self.forgetting_factor + np.dot(uxl, psi)
            self.S_d = (1.0 / self.forgetting_factor) * (self.S_d - np.outer(psi, psi) / den)
            
            # 5. Weight Update
            # w(:,i+1) = w(:,i) + elinha(i) * Sd * uxl(:,i)
            self.w = self.w + e_prior * np.dot(self.S_d, uxl)
            
            # 6. Compute Output and Posterior Error
            y[k] = np.dot(self.w, uxl)
            e[k] = d[k] - y[k]
            
            self._record_history()

        if verbose:
            print(f"Volterra RLS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF