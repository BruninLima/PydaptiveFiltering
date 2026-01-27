#  lms.lms_newton.py
#
#       Implements the Complex LMS-Newton algorithm for COMPLEX valued data.
#       (Algorithm 4.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class LMSNewton(AdaptiveFilter):
    """
    Description
    -----------
    Implements the LMS-Newton algorithm for COMPLEX valued data.
    This algorithm approximates the Newton method by using a recursive estimate 
    of the inverse correlation matrix of the input signal to decorrelate the 
    data and speed up convergence.
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        alpha: float, 
        initial_inv_rx: np.ndarray, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order   : int (The order of the filter M)
            alpha          : float (Forgetting factor 0 < alpha < 1)
            initial_inv_rx : np.ndarray (Initial inverse correlation matrix M+1 x M+1)
            step           : float (Convergence factor mu)
            w_init         : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.alpha: float = alpha
        self.inv_rx: np.ndarray = np.array(initial_inv_rx, dtype=complex)
        self.step: float = step

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the LMS-Newton algorithm.

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
        """
        tic: float = time()
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=complex)
        d_in: np.ndarray = np.asarray(desired_signal, dtype=complex)
        
        self._validate_inputs(x_in, d_in)
        n_iterations: int = d_in.size
        
        self.errors: np.ndarray = np.zeros(n_iterations, dtype=complex)
        self.outputs: np.ndarray = np.zeros(n_iterations, dtype=complex)

        

        for k in range(n_iterations):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]

            self.outputs[k] = np.dot(self.w.conj(), self.regressor)

            self.errors[k] = d_in[k] - self.outputs[k]

            x_vec: np.ndarray = self.regressor.reshape(-1, 1)
            x_h: np.ndarray = x_vec.conj().T
            
            phi = (x_h @ self.inv_rx @ x_vec).item()
            aux_den = ((1.0 - self.alpha) / self.alpha) + phi
            
            num = (self.inv_rx @ x_vec) @ (x_h @ self.inv_rx)
            
            self.inv_rx = (self.inv_rx - (num / aux_den)) / (1.0 - self.alpha)

            update_vector = self.step * self.errors[k].conj() * (self.inv_rx @ self.regressor)
            self.w = self.w + update_vector
            
            self._record_history()

        if verbose:
            runtime: float = (time() - tic) * 1000
            print(f"[LMS-Newton] Adaptation completed in {runtime:.3f} ms.")

        return {
            'outputs': self.outputs,
            'errors': self.errors,
            'coefficients': self.w_history
        }
# EOF