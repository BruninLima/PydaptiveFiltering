#  lms.lms.py
#
#       Implements the Complex LMS algorithm for COMPLEX valued data.
#       (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical
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

class LMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Complex LMS algorithm for COMPLEX valued data.
        (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (The order of the filter M)
            step         : float (Convergence factor mu)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
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
            Executes the weight update process for the LMS algorithm.

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

        Main Variables
        --------- 
            regressor      : Vector containing the tapped delay line (x_k).
            outputs_vector : Represents the output at iteration k (y).
            error_vector   : Represents the output errors at iteration k (e).

        Misc Variables
        --------------
            tic            : Initial time for runtime calculation.
            n_samples      : Number of iterations based on signal size.

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
        
        x: np.ndarray = np.asarray(input_signal, dtype=complex)
        d: np.ndarray = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=complex)
        x_padded[self.m:] = x

        

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.m + 1][::-1]
            
            y[k] = np.dot(self.w.conj(), x_k)
            
            e[k] = d[k] - y[k]

            self.w = self.w + self.step * np.conj(e[k]) * x_k
            
            self._record_history()

        if verbose:
            print(f"LMS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF