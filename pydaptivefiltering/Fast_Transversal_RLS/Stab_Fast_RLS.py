#  RLS.StabFastRLS.py
#
#       Implements the Stabilized Fast Transversal RLS algorithm for COMPLEX valued data.
#       (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical
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
from pydaptivefiltering.main import AdaptiveFilter

class StabFastRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Stabilized Fast Transversal RLS algorithm for COMPLEX valued data.
        (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """

    def __init__(
        self, 
        filter_order: int, 
        lamb: float = 0.99, 
        epsilon: float = 0.1,
        kappa1: float = 1.5,
        kappa2: float = 2.5,
        kappa3: float = 1.0,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (The order of the filter M)
            lamb         : float (Forgetting factor lambda, 0 << lambda <= 1)
            epsilon      : float (Initialization of xiMin_backward and xiMin_forward)
            kappa1,2,3   : float (Stabilization constants)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.lamb: float = lamb
        self.epsilon: float = epsilon
        self.kappa1: float = kappa1
        self.kappa2: float = kappa2
        self.kappa3: float = kappa3

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Stabilized Fast RLS algorithm.

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
            gamma_n_3      : Conversion factor.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins         - wallace.wam@gmail.com      & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
        """
        tic = time()
        x = np.asarray(input_signal, dtype=complex)
        d = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x, d)
        n_samples = x.size
        m = self.m + 1 
        
        y_priori = np.zeros(n_samples, dtype=complex)
        e_priori = np.zeros(n_samples, dtype=complex)
        y_post = np.zeros(n_samples, dtype=complex)
        e_post = np.zeros(n_samples, dtype=complex)

        w_f = np.zeros(m, dtype=complex)
        w_b = np.zeros(m, dtype=complex)
        phi_hat_n = np.zeros(m, dtype=complex)
        gamma_n = 1.0  
        xi_min_f = self.epsilon
        xi_min_b = self.epsilon

        x_padded = np.zeros(n_samples + m, dtype=complex)
        x_padded[m:] = x

        for k in range(n_samples):
            reg_m_plus_1 = x_padded[k : k + m + 1][::-1]
            reg_m = reg_m_plus_1[:-1]
            
            e_f_line = reg_m_plus_1[0] - np.dot(w_f.conj(), reg_m_plus_1[1:])
            e_f = e_f_line * gamma_n
            
            phi_gain = e_f_line / (self.lamb * xi_min_f + 1e-12)
            phi_hat_np1 = np.zeros(m + 1, dtype=complex)
            phi_hat_np1[0] = phi_gain
            phi_hat_np1[1:] = phi_hat_n - phi_gain * w_f
            
            gamma_np1 = 1.0 / np.real((1.0 / gamma_n) + phi_hat_np1[0] * np.conj(e_f_line) + 1e-12)
            gamma_np1 = np.clip(gamma_np1, 1e-6, 1.0)
            
            xi_min_f = self.lamb * xi_min_f + np.real(e_f * np.conj(e_f_line))

            w_f = w_f + phi_hat_n * np.conj(e_f)
            
            e_b_line_1 = self.lamb * xi_min_b * phi_hat_np1[-1]
            e_b_line_2 = reg_m_plus_1[-1] - np.dot(w_b.conj(), reg_m_plus_1[:-1])
            
            e_b_line_v1 = e_b_line_2 * self.kappa1 + e_b_line_1 * (1.0 - self.kappa1)
            e_b_line_v2 = e_b_line_2 * self.kappa2 + e_b_line_1 * (1.0 - self.kappa2)
            e_b_line_v3 = e_b_line_2 * self.kappa3 + e_b_line_1 * (1.0 - self.kappa3)
            
            denom_gamma = (1.0 / gamma_np1) - np.real(phi_hat_np1[-1] * np.conj(e_b_line_v3))
            gamma_n_new = 1.0 / (denom_gamma + 1e-12)
            gamma_n_new = np.clip(gamma_n_new, 1e-6, 1.0)
            
            e_b_v1 = e_b_line_v1 * gamma_n_new
            e_b_v2 = e_b_line_v2 * gamma_n_new
            
            xi_min_b = self.lamb * xi_min_b + np.real(e_b_v2 * np.conj(e_b_line_v2))
            phi_hat_n = phi_hat_np1[:-1] + phi_hat_np1[-1] * w_b
            w_b = w_b + phi_hat_n * np.conj(e_b_v1)
            
            gamma_n = gamma_n_new
            
            y_priori[k] = np.dot(self.w.conj(), reg_m)
            e_priori[k] = d[k] - y_priori[k]
            
            e_post[k] = e_priori[k] * gamma_n
            y_post[k] = d[k] - e_post[k]
            
            self.w = self.w + phi_hat_n * np.conj(e_post[k])
            
            self._record_history()

        if verbose:
            print(f"StabFastRLS completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y_priori, 'errors': e_priori, 'coefficients': self.w_history,
            'outputs_posteriori': y_post, 'errors_posteriori': e_post
        }


# EOF