#  LMS.SignError.py
#
#       Implements the Sign-Error LMS algorithm for Real valued data.
#       (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
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
from pydaptivefiltering.main import AdaptiveFilter

class SignError(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Sign-Error LMS algorithm for Real valued data.
        (Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """

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
            Executes the weight update process for the Sign-Error LMS algorithm.

        Inputs
        -------
            desired_signal : numpy array (row vector)
            input_signal   : numpy array (row vector)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output of each iteration.
                errors       : Store the error for each iteration.
                coefficients : Store the estimated coefficients for each iteration.

        Main Variables
        ---------
            regressor      : Vector containing the tapped delay line.
            outputs_vector : Represents the output at iteration k.
            error_vector   : Represents the output errors at iteration k.

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
        
        x_in = np.asarray(input_signal, dtype=float)
        d_in = np.asarray(desired_signal, dtype=float)

        self.w = self.w.real.astype(float)
        self.regressor = self.regressor.real.astype(float)

        self._validate_inputs(x_in, d_in)
        n_samples: int = x_in.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=x_in.dtype)
        e: np.ndarray = np.zeros(n_samples, dtype=x_in.dtype)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=x_in.dtype)
        x_padded[self.m:] = x_in

        

        for k in range(n_samples):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]
            
            y[k] = np.dot(self.w, self.regressor)
            e[k] = d_in[k] - y[k]
            
            self.w += self.step * np.sign(e[k]) * self.regressor
            
            self._record_history()

        if verbose:
            print(f"Sign-Error LMS completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF