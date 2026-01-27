#  rls.rls.py
#
#       Implements the RLS algorithm for COMPLEX valued data.
#       (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

#Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class RLS(AdaptiveFilter):
    """
    Description
    -----------
    Implements the RLS (Recursive Least Squares) algorithm for COMPLEX valued data.
    This algorithm minimizes the weighted least squares objective function, 
    providing fast convergence by using the Matrix Inversion Lemma to update 
    the inverse of the deterministic autocorrelation matrix.
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        delta: float, 
        lamb: float, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (The order of the filter M)
            delta        : float (Initial value for S_d initialization: S_d(0) = delta^-1 * I)
            lamb         : float (Forgetting factor 0 < lamb <= 1)
            w_init       : array_like, optional (Initial coefficients vector)
        """
        super().__init__(filter_order, w_init)
        self.lamb: float = lamb
        self.delta: float = delta
        
        self.S_d: np.ndarray = (1.0 / delta) * np.eye(self.m + 1, dtype=complex)
        
        self.p_d: np.ndarray = np.zeros(self.m + 1, dtype=complex)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the RLS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs            : Estimated output using a priori weights.
                errors             : A priori estimation error.
                outputs_posteriori : Estimated output using updated weights.
                errors_posteriori  : A posteriori estimation error.
                coefficients       : History of estimated coefficients.

        Main Variables
        --------- 
            regressor : Vector containing the tapped delay line (x_k).
            S_d       : Inverse of the deterministic autocorrelation matrix estimate.
            p_d       : Estimate of the deterministic cross-correlation vector.
            psi       : Auxiliary scalar (x^H * S_d * x).
            den       : Auxiliary denominator for S_d update (lambda + psi).
            num       : Auxiliary matrix for S_d update (S_d * x * x^H * S_d).

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=complex)
        d_in: np.ndarray = np.asarray(desired_signal, dtype=complex)
        self._validate_inputs(x_in, d_in)
        
        n_iterations: int = d_in.size
        
        self.outputs = np.zeros(n_iterations, dtype=complex)
        self.errors = np.zeros(n_iterations, dtype=complex)
        self.outputs_post = np.zeros(n_iterations, dtype=complex)
        self.errors_post = np.zeros(n_iterations, dtype=complex)

        

        for k in range(n_iterations):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]
            
            self.outputs[k] = np.dot(self.w.conj(), self.regressor)
            self.errors[k] = d_in[k] - self.outputs[k]

            x_vec = self.regressor.reshape(-1, 1)
            x_h = x_vec.conj().T
            
            psi = (x_h @ self.S_d @ x_vec).item()
            den = self.lamb + psi
            
            num = (self.S_d @ x_vec) @ (x_h @ self.S_d)
            
            self.S_d = (1.0 / self.lamb) * (self.S_d - (num / den))
            
            self.p_d = self.lamb * self.p_d + d_in[k].conj() * self.regressor

            self.w = self.S_d @ self.p_d

            self.outputs_post[k] = np.dot(self.w.conj(), self.regressor)
            self.errors_post[k] = d_in[k] - self.outputs_post[k]
            
            self._record_history()

        if verbose:
            runtime: float = (time() - tic) * 1000
            print(f"[RLS] Adaptation completed in {runtime:.3f} ms.")

        return {
            'outputs': self.outputs,
            'errors': self.errors,
            'outputs_posteriori': self.outputs_post,
            'errors_posteriori': self.errors_post,
            'coefficients': self.w_history
        }
# EOF