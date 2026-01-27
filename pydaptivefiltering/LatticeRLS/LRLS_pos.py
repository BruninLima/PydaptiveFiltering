# LatticeRLS.LRLS_pos.py
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
#       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict
from pydaptivefiltering.main import AdaptiveFilter

class LatticeRLS(AdaptiveFilter):
    """
    Lattice Recursive Least Squares (LRLS) algorithm using a posteriori errors.

    This class implements the LRLS algorithm as described in Algorithm 7.1 of the 
    book "Adaptive Filtering: Algorithms and Practical Implementation" by 
    Paulo S. R. Diniz. The lattice structure provides a computationally efficient 
    orthogonalization of the input signal, leading to faster convergence and better 
    numerical stability compared to standard RLS in certain scenarios.

    Attributes:
        filter_order (int): The number of lattice sections (M).
        lam (float): Forgetting factor (0 < lambda <= 1).
        epsilon (float): Small positive constant for energy initialization.
        delta (np.ndarray): Time update of cross-correlation between prediction errors.
        xi_f (np.ndarray): Forward prediction error energy (squared norm).
        xi_b (np.ndarray): Backward prediction error energy (squared norm).
        v (np.ndarray): Ladder coefficients for the joint process estimation.
        delta_v (np.ndarray): Cross-correlation between backward errors and filtering error.
    """

    def __init__(
        self, 
        filter_order: int, 
        lambda_factor: float = 0.99, 
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Initializes the LatticeRLS filter.

        Args:
            filter_order (int): Order of the filter (number of sections).
            lambda_factor (float): Forgetting factor. Defaults to 0.99.
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
        Executes the weight update process for the Lattice RLS algorithm.

        This method processes the input and desired signals sample by sample, 
        performing the lattice prediction stage followed by the joint process 
        estimation (ladder) to compute the filter output and update coefficients.

        Args:
            input_signal (np.ndarray | list): Input signal vector.
            desired_signal (np.ndarray | list): Desired signal vector.
            verbose (bool): If True, prints the execution time. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - 'outputs': The estimated output signal y[k].
                - 'errors': The estimation error signal e[k].
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
            curr_err_b = np.zeros(self.n_sections + 1, dtype=complex)
            curr_err_b[0] = x_in[k]
            

            self.xi_f[0] = self.lam * self.xi_f[0] + np.real(err_f * np.conj(err_f))
            self.xi_b[0] = self.xi_f[0]
            
            gamma_m = 1.0 
            
            for m in range(self.n_sections):

                self.delta[m] = self.lam * self.delta[m] + \
                                (self.error_b_prev[m] * np.conj(err_f)) / gamma_m
                

                kappa_f = np.conj(self.delta[m]) / (self.xi_b[m] + 1e-12)
                kappa_b = self.delta[m] / (self.xi_f[m] + 1e-12)

                new_err_f = err_f - kappa_f * self.error_b_prev[m]
                curr_err_b[m+1] = self.error_b_prev[m] - kappa_b * err_f

                self.xi_f[m+1] = self.xi_f[m] - np.real(kappa_f * self.delta[m])
                self.xi_b[m+1] = self.xi_b[m] - np.real(kappa_b * np.conj(self.delta[m]))

                gamma_m_next = gamma_m - (np.real(curr_err_b[m] * np.conj(curr_err_b[m])) / (self.xi_b[m] + 1e-12))
                
                err_f = new_err_f
                gamma_m = gamma_m_next

            e_posteriori = d_in[k]
            gamma_ladder = 1.0
            
            for m in range(self.n_sections + 1):

                self.delta_v[m] = self.lam * self.delta_v[m] + \
                                  (curr_err_b[m] * np.conj(e_posteriori)) / gamma_ladder
                
                self.v[m] = self.delta_v[m] / (self.xi_b[m] + 1e-12)
                
                e_posteriori = e_posteriori - np.conj(self.v[m]) * curr_err_b[m]
                
                gamma_ladder = gamma_ladder - (np.real(curr_err_b[m] * np.conj(curr_err_b[m])) / (self.xi_b[m] + 1e-12))

            y[k] = d_in[k] - e_posteriori
            e[k] = e_posteriori

            self.error_b_prev = curr_err_b
            
        if verbose:
            print(f"[LatticeRLS] Completed in {(time() - tic)*1000:.02f} ms")
            
        return {'outputs': y, 'errors': e}

# EOF