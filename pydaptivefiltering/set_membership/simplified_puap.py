#  SetMembership.Simp_PUAP.py
#
#       Implements the Simplified Set-membership Partial-Update 
#       Affine-Projection (SM-Simp-PUAP) algorithm for COMPLEX valued data.
#       (Algorithm 6.6 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

#Imports
import warnings
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class SimplifiedSMPUAP(AdaptiveFilter):
    """
    Description
    -----------
    Implements the Simplified Set-membership Partial-Update Affine-Projection (SM-Simp-PUAP) 
    algorithm for COMPLEX valued data.
    (Algorithm 6.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        gamma_bar: float, 
        gamma: float, 
        L: int,
        up_selector: np.ndarray,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order     : int (Filter order M)
            gamma_bar        : float (Upper bound for the error modulus)
            gamma            : float (Regularization factor)
            L                : int (Reuse data factor / constraint length)
            up_selector      : np.ndarray (Matrix M+1 x nIterations with 0 or 1)
            w_init           : array_like, optional (Initial weights)
        """
        
        warnings.warn(
            "SM_Simp_PUAP is currently under development and may not produce intended results. "
            "Instability or divergence (High MSE) has been observed in complex-valued simulations.",
            UserWarning
        )

        super().__init__(filter_order, w_init)
        self.gamma_bar = gamma_bar
        self.gamma = gamma
        self.L = L
        self.up_selector = up_selector
        self.n_updates = 0
        self.regressor_matrix = np.zeros((self.m + 1, self.L + 1), dtype=complex)
        self.X_matrix = self.regressor_matrix

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], int]]:
        """
        Description
        -----------
            Executes the optimization process for the SM-Simp-PUAP algorithm for COMPLEX valued data.
            (Algorithm 6.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

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
                n_updates    : Number of filter coefficient updates.

        Main Variables
        --------- 
            regressor_matrix : Matrix containing current and past input vectors.
            error_k          : Scalar a priori error (conjugate).
            n_updates        : Count of iterations where |e(k)| > gamma_bar.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic = time()
        x_in = np.asarray(input_signal, dtype=complex)
        d_in = np.asarray(desired_signal, dtype=complex)
        self._validate_inputs(x_in, d_in)

        n_iterations = d_in.size
        n_coeffs = self.m + 1
        
        self.outputs = np.zeros(n_iterations, dtype=complex)
        self.errors = np.zeros(n_iterations, dtype=complex)
        self.n_updates = 0

        prefixed_input = np.concatenate([np.zeros(n_coeffs - 1, dtype=complex), x_in])
        prefixed_desired = np.concatenate([np.zeros(self.L, dtype=complex), d_in])
        
        w_current = self.w.reshape(-1, 1)
        u1 = np.zeros((self.L + 1, 1), dtype=complex)
        u1[0, 0] = 1.0 

        for it in range(n_iterations):
            self.regressor_matrix[:, 1:] = self.regressor_matrix[:, :-1]
            start_idx = it + n_coeffs - 1
            self.regressor_matrix[:, 0] = prefixed_input[start_idx : (it - 1 if it > 0 else None) : -1]

            output_ap_conj = (self.regressor_matrix.conj().T) @ w_current
            desired_slice = prefixed_desired[it + self.L : (it - 1 if it > 0 else None) : -1]
            error_ap_conj = desired_slice.conj().reshape(-1, 1) - output_ap_conj

            error_k = error_ap_conj[0, 0]
            if np.abs(error_k) > self.gamma_bar:
                self.n_updates += 1
                mu = 1 - (self.gamma_bar / np.abs(error_k))
            else:
                mu = 0.0

            c_vec = self.up_selector[:, it].reshape(-1, 1)
            
            if mu > 0:
                C_reg = c_vec * self.regressor_matrix
                
                R = (self.regressor_matrix.conj().T @ C_reg) + self.gamma * np.eye(self.L + 1)
                
                try:
                    inv_term = np.linalg.solve(R, mu * error_k * u1)
                    w_current = w_current + (C_reg @ inv_term)
                except np.linalg.LinAlgError:
                    w_current = w_current + (C_reg @ (np.linalg.pinv(R) @ (mu * error_k * u1)))

            self.outputs[it] = output_ap_conj[0, 0].item()
            self.errors[it] = error_k.item()
            self.w = w_current.flatten()
            self._record_history()

        if verbose:
            runtime = (time() - tic) * 1000
            print(f"[SM-Simp-PUAP] Updates: {self.n_updates}/{n_iterations} | Runtime: {runtime:.2f} ms")

        return {
            'outputs': self.outputs, 'errors': self.errors,
            'coefficients': self.w_history, 'n_updates': self.n_updates
        }
# EOF