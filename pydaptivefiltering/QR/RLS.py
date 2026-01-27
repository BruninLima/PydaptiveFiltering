# QR_RLS.py
#
#       Implements the QR-RLS algorithm for COMPLEX valued data.
#       (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom            & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com      & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.main import AdaptiveFilter

class QR_RLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the QR-RLS algorithm for COMPLEX valued data.
        (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """

    def __init__(
        self, 
        filter_order: int, 
        lamb: float = 0.99, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (The order of the filter M)
            lamb         : float (Forgetting factor lambda, 0 << lamb <= 1)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.lamb: float = lamb
        self.n_coeffs: int = self.m + 1
        
        self.u_line: np.ndarray = np.zeros((self.n_coeffs, self.n_coeffs), dtype=complex)
        self.d_line_q2: np.ndarray = np.zeros(self.n_coeffs, dtype=complex)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the QR-RLS algorithm using Givens Rotations.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose         : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs            : Store the estimated output of each iteration (a priori).
                errors             : Store the error for each iteration (a priori).
                coefficients       : Store the estimated coefficients for each iteration.
                errors_posteriori  : Store the a posteriori error for each iteration.

        Main Variables
        --------- 
            u_line         : Upper triangular matrix (R matrix from QR).
            d_line_q2      : Transformed desired signal vector.
            gamma          : Likelihood variable.
            regressor      : Vector containing the tapped delay line (x_k).

        Misc Variables
        --------------
            tic            : Initial time for runtime calculation.
            n_samples      : Number of iterations based on signal size.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom            & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com      & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal, dtype=complex)
        d: np.ndarray = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)
        e_post: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=complex)
        x_padded[self.m:] = x

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.n_coeffs][::-1]
            
            y[k] = np.dot(self.w.conj(), x_k)
            e[k] = d[k] - y[k]

            gamma = 1.0 + 0j
            d_prime = d[k]
            regressor_row = x_k.copy()
            
            self.u_line *= np.sqrt(self.lamb)
            self.d_line_q2 *= np.sqrt(self.lamb)
            
            for i in range(self.n_coeffs):
                u_ii = self.u_line[i, i]
                x_i = regressor_row[i]
                
                norm = np.sqrt(np.abs(u_ii)**2 + np.abs(x_i)**2)
                
                if norm > 1e-18:
                    c = u_ii / norm
                    s = x_i / norm
                else:
                    c = 1.0 + 0j
                    s = 0.0 + 0j
                
                for j in range(i, self.n_coeffs):
                    u_old = self.u_line[i, j]
                    x_old = regressor_row[j]
                    
                    self.u_line[i, j] = np.conj(c) * u_old + np.conj(s) * x_old
                    regressor_row[j] = -s * u_old + c * x_old
                
                d_q2_old = self.d_line_q2[i]
                self.d_line_q2[i] = np.conj(c) * d_q2_old + np.conj(s) * d_prime
                d_prime = -s * d_q2_old + c * d_prime
                
                gamma *= c

            w_new = np.zeros(self.n_coeffs, dtype=complex)
            for i in range(self.n_coeffs - 1, -1, -1):
                if np.abs(self.u_line[i, i]) > 1e-18:
                    sum_val = np.dot(self.u_line[i, i+1:], w_new[i+1:])
                    w_new[i] = (self.d_line_q2[i] - sum_val) / self.u_line[i, i]
            
            self.w = w_new
            
            e_post[k] = d_prime * np.conj(gamma)
            
            self._record_history()

        if verbose:
            print(f"QR-RLS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history,
            'errors_posteriori': e_post
        }

# EOF