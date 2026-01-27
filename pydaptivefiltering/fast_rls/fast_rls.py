#  fast_rls.fast_rls.py
#
#       Implements the Fast Transversal RLS algorithm for COMPLEX valued data.
#       (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins         - wallace.wam@gmail.com      & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class FastRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Fast Transversal RLS algorithm for COMPLEX valued data.
        (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        forgetting_factor: float = 0.99, 
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order                      : int (The order of the filter M)
            forgetting_factor   (lambda)      : float (Forgetting factor lambda, 0 << lambda <= 1)
            epsilon                           : float (Initialization of xiMin_backward and xiMin_forward)
            w_init                            : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.forgetting_factor: float = forgetting_factor
        self.epsilon: float = epsilon

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Fast RLS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs            : Store the estimated output (priori) of each iteration.
                errors             : Store the priori error for each iteration.
                coefficients       : Store the estimated coefficients for each iteration.
                outputs_posteriori : Store the a posteriori estimated output.
                errors_posteriori  : Store the a posteriori error.

        Main Variables
        --------- 
            w_f, w_b       : Forward and backward predictor coefficients.
            xi_min_f, b    : Minimum forward and backward squared errors.
            phi_hat_n      : Gain vector.
            gamma_n        : Conversion factor.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins         - wallace.wam@gmail.com      & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal, dtype=complex)
        d: np.ndarray = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x, d)
        n_samples: int = x.size
        m_plus_1: int = self.m + 1 
        
        y_priori: np.ndarray = np.zeros(n_samples, dtype=complex)
        e_priori: np.ndarray = np.zeros(n_samples, dtype=complex)
        y_post: np.ndarray = np.zeros(n_samples, dtype=complex)
        e_post: np.ndarray = np.zeros(n_samples, dtype=complex)

        w_f: np.ndarray = np.zeros(m_plus_1, dtype=complex)
        w_b: np.ndarray = np.zeros(m_plus_1, dtype=complex)
        phi_hat_n: np.ndarray = np.zeros(m_plus_1, dtype=complex)
        gamma_n: float = 1.0
        xi_min_f_prev: float = self.epsilon
        xi_min_b: float = self.epsilon

        x_padded: np.ndarray = np.zeros(n_samples + m_plus_1, dtype=complex)
        x_padded[m_plus_1:] = x

        for k in range(n_samples):
            regressor: np.ndarray = x_padded[k : k + m_plus_1 + 1][::-1]
            
            e_f_priori = regressor[0] - np.dot(w_f.conj(), regressor[1:])
            e_f_post = e_f_priori * gamma_n
            
            xi_min_f_curr = self.forgetting_factor * xi_min_f_prev + e_f_priori * np.conj(e_f_post)
            
            phi_gain = e_f_priori / (self.forgetting_factor * xi_min_f_prev)
            phi_hat_n_plus_1 = np.zeros(m_plus_1 + 1, dtype=complex)
            phi_hat_n_plus_1[1:] = phi_hat_n
            phi_hat_n_plus_1[0] += phi_gain
            phi_hat_n_plus_1[1:] -= phi_gain * w_f
            
            w_f = w_f + phi_hat_n * np.conj(e_f_post)
            
            gamma_n_plus_1 = (self.forgetting_factor * xi_min_f_prev * gamma_n) / xi_min_f_curr
            e_b_priori = self.forgetting_factor * xi_min_b * phi_hat_n_plus_1[-1]
            gamma_n = 1.0 / ( (1.0 / gamma_n_plus_1) - (phi_hat_n_plus_1[-1] * np.conj(e_b_priori)) )
            
            e_b_post = e_b_priori * gamma_n
            xi_min_b = self.forgetting_factor * xi_min_b + e_b_post * np.conj(e_b_priori)
            
            phi_hat_n = phi_hat_n_plus_1[:-1] + phi_hat_n_plus_1[-1] * w_b
            w_b = w_b + phi_hat_n * np.conj(e_b_post)

            y_priori[k] = np.dot(self.w.conj(), regressor[:-1])
            e_priori[k] = d[k] - y_priori[k]
            
            e_post[k] = e_priori[k] * gamma_n
            y_post[k] = d[k] - e_post[k]
            
            self.w = self.w + phi_hat_n * np.conj(e_post[k])
            
            xi_min_f_prev = xi_min_f_curr
            self._record_history()

        if verbose:
            print(f"Fast RLS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y_priori,
            'errors': e_priori,
            'coefficients': self.w_history,
            'outputs_posteriori': y_post,
            'errors_posteriori': e_post
        }
# EOF